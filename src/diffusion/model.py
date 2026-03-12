import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time embedding and group normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        time_dim: int = 256,
        groups: int = 32,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t_emb = self.time_proj(F.silu(t))[:, :, None, None]
        h = h + t_emb

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + residual


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))


class UNet(nn.Module):
    """Compact U-Net for diffusion denoising."""

    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_size = image_size

        self.time_embed = TimeEmbedding(time_dim)

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = ResBlock(base_channels, base_channels, time_dim=time_dim)
        self.downsample1 = Downsample(base_channels)

        self.down2 = ResBlock(base_channels, base_channels * 2, time_dim=time_dim)
        self.downsample2 = Downsample(base_channels * 2)

        self.mid = ResBlock(base_channels * 2, base_channels * 2, time_dim=time_dim)

        self.upsample1 = Upsample(base_channels * 2)
        self.up1 = ResBlock(base_channels * 4, base_channels, time_dim=time_dim)

        self.upsample2 = Upsample(base_channels)
        self.up2 = ResBlock(base_channels * 2, base_channels, time_dim=time_dim)

        self.final_norm = nn.GroupNorm(32, base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        x0 = self.init_conv(x)

        x1 = self.down1(x0, t_emb)
        x = self.downsample1(x1)

        x2 = self.down2(x, t_emb)
        x = self.downsample2(x2)

        x = self.mid(x, t_emb)

        x = self.upsample1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x, t_emb)

        x = self.upsample2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x, t_emb)

        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        return x
