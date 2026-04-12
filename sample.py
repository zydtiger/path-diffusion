"""CLI for generating samples from a trained diffusion checkpoint."""

from pathlib import Path

import torch
import typer
from torchvision.utils import save_image

from diffusion.model import DiT, UNet
from diffusion.sampling import sample
from diffusion.scheduler import LinearNoiseScheduler

app = typer.Typer(add_completion=False)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.command()
def main(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-ckpt"),
    num_samples: int = typer.Option(8, "--num-samples", "-n", min=1),
    output: Path = typer.Option(
        Path("outputs/generated_samples.png"), "--output", "-o"
    ),
    dit: bool = typer.Option(
        False,
        "--dit",
        help="Use DiT backbone. If omitted, checkpoint config is used when available.",
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

    backbone = str(cfg.get("backbone", "unet")).lower()
    use_dit = dit or backbone == "dit"

    if use_dit:
        model = DiT(
            image_size=image_size,
            in_channels=num_channels,
            out_channels=num_channels,
        ).to(device)
    else:
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

    samples = sample(model, scheduler, num_samples=num_samples, device=device)

    output.parent.mkdir(parents=True, exist_ok=True)
    sample_images = ((samples.detach().cpu() + 1) / 2).clamp(0, 1)
    save_image(sample_images, output, nrow=max(1, min(8, num_samples)))

    typer.echo(f"Saved {num_samples} samples to '{output}'")


if __name__ == "__main__":
    app()
