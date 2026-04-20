from pathlib import Path
import math
from typing import Callable

import torch
import typer
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from diffusion.dataset import get_dataset
from diffusion.model import DiT, UNet
from diffusion.scheduler import LinearNoiseScheduler

app = typer.Typer(add_completion=False)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _gaussian_kernel2d(kernel_size: int, sigma: float, device: str, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g1 = torch.exp(-(coords**2) / (2 * sigma**2))
    g1 = g1 / g1.sum()
    kernel2d = torch.outer(g1, g1)
    return kernel2d / kernel2d.sum()


def _gaussian_blur_per_channel(
    x: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    kernel2d = _gaussian_kernel2d(
        kernel_size=kernel_size,
        sigma=sigma,
        device=str(x.device),
        dtype=x.dtype,
    )
    channels = x.shape[1]
    weight = kernel2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    padding = kernel_size // 2
    # Depthwise convolution applies the same blur in each channel, preserving color relationships.
    return F.conv2d(x, weight, padding=padding, groups=channels)


@torch.no_grad()
def proximal_sample(
    model: torch.nn.Module,
    scheduler: LinearNoiseScheduler,
    measurement: torch.Tensor,
    forward_op: Callable[[torch.Tensor], torch.Tensor],
    adjoint_op: Callable[[torch.Tensor], torch.Tensor] | None = None,
    num_candidates_max: int = 8,
    data_consistency_eta: float = 0.0,
    data_consistency_eta_final: float = 0.0,
    device: str = "cpu",
) -> torch.Tensor:
    """DPPS-style sampling for inverse problems with measurement operator A.

    We sample multiple candidates from the DDPM posterior at each reverse step and
    pick the proximal candidate with the smallest measurement-consistency error.
    """
    model.eval()
    num_samples = measurement.size(0)
    if adjoint_op is None:
        adjoint_op = forward_op

    # Initialize from measurement in image space: x_T = sqrt(alpha_bar_T) * y + sqrt(1-alpha_bar_T) * eps.
    alpha_bar_T = scheduler.alphas_cumprod[-1]
    x = (
        torch.sqrt(alpha_bar_T) * measurement
        + torch.sqrt(1 - alpha_bar_T) * torch.randn_like(measurement, device=device)
    )

    for t in reversed(range(scheduler.num_timesteps)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # DDPM posterior parameters p_theta(x_{t-1}|x_t)
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

        # Data consistency schedule (applied to x_t below).
        eta_t = scheduler.alphas[-t] * (
            data_consistency_eta - data_consistency_eta_final
        ) + data_consistency_eta_final

        # posterior mean
        coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
        coef_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        mean = coef_x0 * x0_pred + coef_xt * x

        if t > 0:
            posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            posterior_std = torch.sqrt(posterior_var)

            # DPPS adaptive candidate count based on SNR lambda_t.
            snr_t = float((alpha_bar_t / (1 - alpha_bar_t + 1e-8)).item())
            num_candidates = max(int(num_candidates_max * (1.0 - math.exp(-snr_t))), 2)
            num_candidates = min(max(2, num_candidates), num_candidates_max)

            # Draw x_{t-1}^i = mean + sigma_t * z_i.
            noises = torch.randn(
                (num_candidates, *x.shape), device=x.device, dtype=x.dtype
            )
            candidates = mean.unsqueeze(0) + posterior_std * noises

            # Proximal objective for general A:
            # min_i ||A x_{t-1}^i - (coef_xt * A x_t + coef_x0 * y)||_2^2
            target = coef_xt * forward_op(x) + coef_x0 * measurement
            flat_candidates = candidates.reshape(-1, *x.shape[1:])
            cand_measurements = forward_op(flat_candidates).reshape_as(candidates)
            residual = cand_measurements - target.unsqueeze(0)
            candidate_scores = residual.pow(2).flatten(2).sum(dim=-1)  # [K, B]

            best_idx = candidate_scores.argmin(dim=0)
            batch_idx = torch.arange(num_samples, device=x.device)
            x = candidates[best_idx, batch_idx]
        else:
            x = mean

        if eta_t > 0:
            x = x - eta_t * adjoint_op(forward_op(x) - measurement)
            x = torch.clamp(x, -1.0, 1.0)

        if isinstance(model, DiT):
            x = _gaussian_blur_per_channel(x, kernel_size=3, sigma=0.01)
            x = torch.clamp(x, -1.0, 1.0)

    return torch.clamp(x, -1.0, 1.0)


@torch.no_grad()
def standard_inverse_sample(
    model: torch.nn.Module,
    scheduler: LinearNoiseScheduler,
    measurement: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Standard ancestral reverse process initialized from noisy measurement."""
    model.eval()
    num_samples = measurement.size(0)

    alpha_bar_T = scheduler.alphas_cumprod[-1]
    x = (
        torch.sqrt(alpha_bar_T) * measurement
        + torch.sqrt(1 - alpha_bar_T) * torch.randn_like(measurement, device=device)
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

        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

        coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
        coef_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        mean = coef_x0 * x0_pred + coef_xt * x

        if t > 0:
            posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            x = mean + torch.sqrt(posterior_var) * torch.randn_like(x)
        else:
            x = mean

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

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    # Reserve enough left gutter for the widest label plus clear spacing.
    text_probe = Image.new("RGB", (1, 1), color=(255, 255, 255))
    probe_draw = ImageDraw.Draw(text_probe)
    max_label_w = max(
        (probe_draw.textbbox((0, 0), label, font=font)[2] for label in row_labels),
        default=0,
    )
    left_label_pad = max(140, max_label_w + 28)
    right_pad = 8
    top_pad = 8
    bottom_pad = 8
    row_gap = 4
    col_gap = 2

    grid_w = num_samples * plot_image_size + (num_samples - 1) * col_gap
    grid_h = num_rows * plot_image_size + (num_rows - 1) * row_gap

    canvas_h = top_pad + grid_h + bottom_pad
    canvas_w = int(left_label_pad + grid_w + right_pad)

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
    num_candidates_max: int = typer.Option(
        8,
        "--num-candidates-max",
        min=2,
        help="Maximum number of DPPS candidate samples per reverse step.",
    ),
    blur_sigma: float = typer.Option(
        0.8,
        "--blur-sigma",
        min=0.01,
        help="Gaussian blur sigma used to corrupt ground-truth images.",
    ),
    blur_kernel_size: int = typer.Option(
        5,
        "--blur-kernel-size",
        min=3,
        help="Odd Gaussian kernel size used for blur corruption.",
    ),
    dc_eta: float = typer.Option(
        0.0,
        "--dc-eta",
        min=0.0,
        help="Per-step data-consistency strength for Ax~=y (0 disables extra enforcement).",
    ),
    dc_eta_final: float = typer.Option(
        0.0,
        "--dc-eta-final",
        min=0.0,
        help="Final-step data-consistency strength (cosine-decayed from --dc-eta).",
    ),
    dit: bool = typer.Option(
        False,
        "--dit",
        help="Use DiT backbone. If omitted, checkpoint config is used when available.",
    ),
) -> None:
    device = _auto_device()

    # Keep checkpoint deserialization on CPU and use mmap when available to
    # reduce upfront load overhead.
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
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
        )
    else:
        model = UNet(
            image_size=image_size,
            in_channels=num_channels,
            out_channels=num_channels,
        )
    # assign=True (when supported) can reduce copies and speed up loading.
    try:
        model.load_state_dict(ckpt["model_state_dict"], assign=True)
    except TypeError:
        model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    scheduler = LinearNoiseScheduler(
        num_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        image_size=image_size,
        num_channels=num_channels,
    ).to(device)

    # get GT data from test set
    _, test_loader = get_dataset(batch_size=num_samples, image_size=image_size)
    batch = next(iter(test_loader))
    ground_truth = batch[0].to(device)

    if blur_kernel_size % 2 == 0:
        raise typer.BadParameter("--blur-kernel-size must be odd.")

    # Corrupt the ground truth with per-channel Gaussian blur (no channel mixing).
    noisy_input = _gaussian_blur_per_channel(
        ground_truth,
        kernel_size=blur_kernel_size,
        sigma=blur_sigma,
    )

    def blur_op(z: torch.Tensor) -> torch.Tensor:
        # Keep A linear for stable proximal/data-consistency updates.
        return _gaussian_blur_per_channel(z, kernel_size=blur_kernel_size, sigma=blur_sigma)

    typer.echo("Reconstructing samples with standard inverse sampling...")
    standard_recon = standard_inverse_sample(
        model,
        scheduler,
        noisy_input,
        device=device,
    )

    typer.echo("Reconstructing samples with DPPS proximal sampling...")
    proximal_recon = proximal_sample(
        model,
        scheduler,
        noisy_input,
        forward_op=blur_op,
        adjoint_op=blur_op,
        num_candidates_max=num_candidates_max,
        data_consistency_eta=dc_eta,
        data_consistency_eta_final=dc_eta_final,
        device=device,
    )

    gt_viz = _to_viz(ground_truth)
    noisy_viz = _to_viz(noisy_input)
    standard_viz = _to_viz(standard_recon)
    proximal_viz = _to_viz(proximal_recon)

    _save_labeled_rows(
        rows=[gt_viz, noisy_viz, standard_viz, proximal_viz],
        row_labels=["Original", "Noisy Input", "Standard Recon", "Proximal Sampling Recon"],
        output=output,
        plot_image_size=224,
    )

    typer.echo(f"Saved reconstructed samples to '{output}'")
    typer.echo("Rows: Original, Noisy Input, Standard Recon, Proximal Sampling Recon")


if __name__ == "__main__":
    app()
