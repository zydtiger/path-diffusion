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

        alpha_t = scheduler.alphas[t]
        beta_t = scheduler.betas[t]
        alpha_bar_t = scheduler.alphas_cumprod[t]
        alpha_bar_prev = (
            scheduler.alphas_cumprod_prev[t]
            if t > 0
            else torch.ones((), device=device, dtype=x.dtype)
        )

        # Predict x_0 from the current noisy sample and epsilon estimate.
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(
            alpha_bar_t
        )
        x0_pred = torch.clamp(x0_pred, -1, 1)

        # Posterior mean q(x_{t-1} | x_t, x_0).
        coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar_t)
        coef_xt = (1 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1 - alpha_bar_t)
        model_mean = coef_x0 * x0_pred + coef_xt * x

        if t > 0:
            posterior_variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_variance) * noise
        else:
            x = model_mean

    return torch.clamp(x, -1, 1)
