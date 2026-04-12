from pathlib import Path

import torch
import typer
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from diffusion.dataset import get_dataset
from diffusion.model import UNet
from diffusion.sampling import sample
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
            scheduler.alphas_cumprod_prev[t]
            if t > 0
            else torch.ones((), device=device, dtype=x.dtype)
        )

        # predict x_0
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

        # posterior mean
        coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
        coef_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        mean = coef_x0 * x0_pred + coef_xt * x

        if t > 0:
            posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            noise = torch.randn_like(x)
            x_unknown = mean + torch.sqrt(posterior_var) * noise

            # RePaint conditioning: sample known pixels at the same reverse-step level.
            # x_known ~ N(sqrt(alpha_bar_{t-1}) * x0, (1 - alpha_bar_{t-1}) I)
            x_known = (
                torch.sqrt(alpha_bar_prev) * measurement
                + torch.sqrt(1 - alpha_bar_prev) * noise
            )
        else:
            x_unknown = mean
            x_known = measurement

        x = mask * x_known + (1 - mask) * x_unknown

    return torch.clamp(x, -1.0, 1.0)


def _to_viz(x: torch.Tensor) -> torch.Tensor:
    return ((x.detach().cpu() + 1) / 2).clamp(0, 1)


def _save_labeled_rows(
    rows: list[torch.Tensor],
    row_labels: list[str],
    output: Path,
    plot_image_size: int = 224,
) -> None:
    if not rows:
        raise ValueError("No rows provided for visualization")
    if len(rows) != len(row_labels):
        raise ValueError("rows and row_labels must have the same length")

    num_rows = len(rows)
    num_samples = rows[0].shape[0]
    num_channels = rows[0].shape[1]
    display_rows = [
        F.interpolate(row, size=(plot_image_size, plot_image_size), mode="nearest")
        for row in rows
    ]

    left_label_pad = 140
    right_pad = 8
    top_pad = 8
    bottom_pad = 8
    row_gap = 4
    col_gap = 2

    grid_w = num_samples * plot_image_size + (num_samples - 1) * col_gap
    grid_h = num_rows * plot_image_size + (num_rows - 1) * row_gap

    canvas_h = top_pad + grid_h + bottom_pad
    canvas_w = left_label_pad + grid_w + right_pad

    canvas = torch.ones((num_channels, canvas_h, canvas_w), dtype=rows[0].dtype)

    for row_idx, row in enumerate(display_rows):
        row_start = top_pad + row_idx * (plot_image_size + row_gap)
        for col_idx in range(num_samples):
            col_start = left_label_pad + col_idx * (plot_image_size + col_gap)
            canvas[
                :, row_start : row_start + plot_image_size, col_start : col_start + plot_image_size
            ] = row[col_idx]

    if num_channels == 1:
        pil_img = Image.fromarray((canvas.squeeze(0).numpy() * 255).astype("uint8"), mode="L")
        text_fill = 0
    else:
        hwc = (canvas.permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(hwc)
        text_fill = (0, 0, 0)

    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    for row_idx, label in enumerate(row_labels):
        row_center_y = top_pad + row_idx * (plot_image_size + row_gap) + plot_image_size // 2
        text_box = draw.textbbox((0, 0), label, font=font)
        text_h = text_box[3] - text_box[1]
        draw.text((10, row_center_y - text_h // 2), label, fill=text_fill, font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(output)


@app.command()
def main(
    checkpoint: Path = typer.Option("./diffusion_model.pt", "--checkpoint", "-ckpt"),
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

    typer.echo("Generating standard samples...")
    standard_samples = sample(model, scheduler, num_samples=num_samples, device=device)

    typer.echo("Reconstructing samples with proximal sampling...")
    proximal_recon = proximal_sample(model, scheduler, masked_input, mask, device=device)

    gt_viz = _to_viz(ground_truth)
    masked_viz = _to_viz(masked_input) * mask.cpu() + 0.5 * (1 - mask.cpu())
    standard_viz = _to_viz(standard_samples)
    proximal_viz = _to_viz(proximal_recon)

    _save_labeled_rows(
        rows=[gt_viz, masked_viz, standard_viz, proximal_viz],
        row_labels=["Original", "Masked", "Standard Sampling", "Proximal Sampling"],
        output=output,
        plot_image_size=224,
    )

    typer.echo(f"Saved reconstructed samples to '{output}'")
    typer.echo("Rows: Original, Masked, Standard Sampling, Proximal Sampling")


if __name__ == "__main__":
    app()
