from pathlib import Path

import torch
import typer
from torchvision.utils import save_image

from diffusion.dataset import get_dataset
from diffusion.model import UNet
from diffusion.scheduler import LinearNoiseScheduler

app = typer.Typer(add_completion=False)

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@torch.no_grad()

def proximal_sample(
    model: torch.nn.Module,
    scheduler: LinearNoiseScheduler,
    measurement: torch.Tensor,
    mask: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    num_samples = measurement.size(0)

    x = torch.randn(
        num_samples, 
        scheduler.num_channels,
        scheduler.image_size,
        scheduler.image_size,
        device=device,
    )

    for t in reversed(range(scheduler.num_timesteps)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # unconditional step
        predicted_noise = model(x, t_batch)
        alpha_t = scheduler.alphas[t]
        beta_t = scheduler.betas[t]

        alpha_bar_t = scheduler.alphas_cumprod[t]
        alpha_bar_prev = (
            scheduler.alphas_cumprod[t] if t > 0 else torch.tensor((), device=device, dtype=x.dtype)
        )

        # predict x_0
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # posterior mean
        coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
        coef_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        mean = coef_x0 * x0_pred + coef_xt * x

        if t > 0:
            posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            noise = torch.randn_like(x)
            x_uncond = mean + torch.sqrt(posterior_var) * noise
        else:
            x_uncond = mean
        
        # proximal step
        if t > 0:
            t_prev_batch = torch.full((num_samples,), t - 1, device=device, dtype=torch.long)
            y_noisy, _ = scheduler.q_sample(measurement, t_prev_batch)
        else:
            y_noisy = measurement

        x = mask * y_noisy + (1 - mask) * x_uncond

    return torch.clamp(x, -1.0, 1.0)

@app.command()
def main(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-ckpt"),
    num_samples: int = typer.Option(8, "--num-samples", "-n", min=1),
    output: Path = typer.Option(
        Path("outputs/reconstructed_samples.png"), "--output", "-o"
    ),
) -> None:
    device = _auto_device()

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    image_size = int(cfg.get("image_size", 28))
    num_timesteps = int(cfg.get("num_timesteps", 1000))
    num_channels = int(cfg.get("num_channels", 3))

    scheduler_betas = ckpt.get("scheduler_betas")
    if scheduler_betas is not None:
        betas = scheduler_betas.detach().cpu().float()
        beta_start = float(betas[0].item())
        beta_end = float(betas[-1].item())
        num_timesteps = int(betas.numel())
    else:
        beta_start = 1e-4
        beta_end = 0.02

    model = UNet(
        image_size=image_size,
        in_channels=num_channels,
        out_channels=num_channels,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    scheduler = LinearNoiseScheduler(
        num_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        image_size=image_size,
        num_channels=num_channels,
    ).to(device)

    # get GT data from test set
    _, test_loader = get_dataset(batch_size=num_samples)
    batch = next(iter(test_loader))
    ground_truth = batch[0].to(device)

    # forward operator
    # "corrupt" data by masking out square in middle
    mask = torch.ones_like(ground_truth)
    mask[:, :, 7:21, 7:21] = 0.0
    masked_input = mask * ground_truth

    # reconstruct using proximal sampling
    typer.echo("Reconstructing samples...")
    reconstructed = proximal_sample(model, scheduler, masked_input, mask, device=device)

    output.parent.mkdir(parents=True, exist_ok=True)
    
    gt_viz = ((ground_truth.detach().cpu() + 1) / 2).clamp(0, 1)

    masked_viz = ((masked_input.detach().cpu() + 1) / 2).clamp(0, 1) * mask.cpu() + 0.5 * (1 - mask.cpu())
    recon_viz = ((reconstructed.detach().cpu() + 1) / 2).clamp(0, 1)

    comparison = torch.stack([gt_viz, masked_viz, recon_viz], dim=0).view(-1, num_channels, image_size, image_size)

    save_image(comparison, output, nrow=3)

    typer.echo(f"Saved reconstructed samples to '{output}'")
    typer.echo("Left: Ground Truth, Middle: Masked Input, Right: Reconstruction")

if __name__ == "__main__":
    app()