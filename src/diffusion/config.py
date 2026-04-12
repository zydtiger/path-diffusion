from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Config:
    """Training and model configuration."""

    # Dataset
    image_size: int = 28
    num_channels: int = 3
    num_classes: int = 7

    # Diffusion
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 50
    ema_decay: float = 0.999

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
