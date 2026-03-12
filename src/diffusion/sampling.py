import torch

from diffusion.scheduler import LinearNoiseScheduler


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    scheduler: LinearNoiseScheduler,
    num_samples: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate samples using reverse diffusion."""
    model.eval()

    x = torch.randn(
        num_samples,
        scheduler.num_channels,
        scheduler.image_size,
        scheduler.image_size,
        device=device,
    )

    for t in reversed(range(scheduler.num_timesteps)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)

        alpha = scheduler.alphas[t]
        beta = scheduler.betas[t]

        if t > 0:
            sigma = torch.sqrt(
                beta
                * (1 - scheduler.alphas_cumprod[t - 1])
                / (1 - scheduler.alphas_cumprod[t])
            )
            noise = torch.randn_like(x)
        else:
            sigma = torch.zeros_like(x)
            noise = 0

        x = (1 / torch.sqrt(alpha)) * (
            x - beta / torch.sqrt(1 - scheduler.alphas_cumprod[t]) * predicted_noise
        )
        x = x - sigma * noise

    return torch.clamp(x, -1, 1)
