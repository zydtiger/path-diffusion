"""Entry point for PathMNIST diffusion training."""

from pathlib import Path
import sys

import torch

# Ensure local `src` package imports work when running `python main.py`.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from diffusion.config import Config
from diffusion.dataset import get_dataset
from diffusion.model import UNet
from diffusion.sampling import sample
from diffusion.scheduler import LinearNoiseScheduler
from diffusion.training import train


def main() -> tuple[torch.nn.Module, LinearNoiseScheduler]:
    config = Config()

    print("=== PathMNIST Diffusion Training ===")
    print(f"Device: {config.device}")

    print("Loading PathMNIST dataset...")
    train_loader, test_loader = get_dataset(batch_size=config.batch_size)
    print(f"Training samples: {len(train_loader.dataset)}") # type: ignore
    print(f"Test samples: {len(test_loader.dataset)}") # type: ignore

    print("Creating U-Net model...")
    model = UNet(
        image_size=config.image_size,
        in_channels=config.num_channels,
        out_channels=config.num_channels,
    ).to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    scheduler = LinearNoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        image_size=config.image_size,
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
    )

    print("\nGenerating samples...")
    samples = sample(model, scheduler, num_samples=8, device=config.device)
    print(f"Generated {samples.size(0)} samples of shape {samples.shape}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scheduler_betas": scheduler.betas,
            "config": {
                "image_size": config.image_size,
                "num_timesteps": config.num_timesteps,
                "num_channels": config.num_channels,
            },
        },
        "diffusion_model.pt",
    )
    print("Model saved to 'diffusion_model.pt'")

    return model, scheduler


if __name__ == "__main__":
    main()
