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


class DiTBlock(nn.Module):
    """Transformer block conditioned on diffusion timestep embeddings."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        time_dim: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.time_to_attn = nn.Linear(time_dim, hidden_size)
        self.time_to_mlp = nn.Linear(time_dim, hidden_size)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x) + self.time_to_attn(F.silu(t_emb))[:, None, :]
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out

        mlp_in = self.norm2(x) + self.time_to_mlp(F.silu(t_emb))[:, None, :]
        x = x + self.mlp(mlp_in)
        return x


class DiT(nn.Module):
    """Compact DiT backbone with patchify-transformer-unpatchify flow."""

    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        out_channels: int = 1,
        patch_size: int = 4,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}"
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        grid_size = image_size // patch_size
        self.num_patches = grid_size * grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        self.blocks = nn.ModuleList(
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                time_dim=time_dim,
            )
            for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.final_proj = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        t_emb = self.time_mlp(self.time_embed(t))

        x = self.patch_embed(x)
        h_patches, w_patches = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, : x.size(1)]

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.final_norm(x)
        x = self.final_proj(x)

        x = x.view(
            batch_size,
            h_patches,
            w_patches,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(
            batch_size,
            self.out_channels,
            h_patches * self.patch_size,
            w_patches * self.patch_size,
        )
        return x
