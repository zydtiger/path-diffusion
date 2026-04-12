"""Entry point for PathMNIST diffusion training."""

from pathlib import Path

import torch
import typer
from torchvision.utils import save_image

from diffusion.config import Config
from diffusion.dataset import get_dataset
from diffusion.model import DiT, UNet
from diffusion.sampling import sample
from diffusion.scheduler import LinearNoiseScheduler
from diffusion.training import train

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dit: bool = typer.Option(
        False,
        "--dit",
        help="Use a DiT backbone instead of U-Net.",
    ),
    large: bool = typer.Option(
        False,
        "--large",
        help="Use MedMNIST+ resolution (224x224) instead of 28x28.",
    ),
) -> tuple[torch.nn.Module, LinearNoiseScheduler]:
    config = Config()
    image_size = 224 if large else config.image_size

    print("=== PathMNIST Diffusion Training ===")
    print(f"Device: {config.device}")

    dataset_label = "PathMNIST+" if large else "PathMNIST"
    print(f"Loading {dataset_label} dataset ({image_size}x{image_size})...")
    train_loader, test_loader = get_dataset(
        batch_size=config.batch_size, image_size=image_size
    )
    print(f"Training samples: {len(train_loader.dataset)}")  # type: ignore
    print(f"Test samples: {len(test_loader.dataset)}")  # type: ignore

    if dit:
        print("Creating DiT model...")
        model = DiT(
            image_size=image_size,
            in_channels=config.num_channels,
            out_channels=config.num_channels,
        ).to(config.device)
    else:
        print("Creating U-Net model...")
        model = UNet(
            image_size=image_size,
            in_channels=config.num_channels,
            out_channels=config.num_channels,
        ).to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = LinearNoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        image_size=image_size,
        num_channels=config.num_channels,
    ).to(config.device)

    print(f"\nTraining for {config.num_epochs} epochs...")
    model = train(
        model,
        train_loader,
        scheduler,
        config.device,
        num_epochs=config.num_epochs,
        lr=config.learning_rate,
        ema_decay=config.ema_decay,
    )

    print("\nGenerating samples...")
    samples = sample(model, scheduler, num_samples=8, device=config.device)
    print(f"Generated {samples.size(0)} samples of shape {samples.shape}")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert from [-1, 1] to [0, 1] before saving.
    sample_images = ((samples.detach().cpu() + 1) / 2).clamp(0, 1)
    save_image(sample_images, output_dir / "generated_samples.png", nrow=4)
    print(f"Saved sample grid to '{output_dir / 'generated_samples.png'}'")

    checkpoint_path = "diffusion_model_dit.pt" if dit else "diffusion_model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scheduler_betas": scheduler.betas,
            "config": {
                "image_size": image_size,
                "num_timesteps": config.num_timesteps,
                "num_channels": config.num_channels,
                "backbone": "dit" if dit else "unet",
            },
        },
        checkpoint_path,
    )
    print(f"Model saved to '{checkpoint_path}'")

    return model, scheduler


if __name__ == "__main__":
    app()
