import torch
from torch import nn


def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_avg = x.mean(dim=1, keepdim=True)
        x_max, _ = x.max(dim=1, keepdim=True)
        return self.conv(torch.cat([x_avg, x_max], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ca(self.gap(x))


class PixelAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2 * dim, dim, kernel_size=7, padding=3, padding_mode="reflect", groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, pattn: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([x, pattn], dim=1)
        return self.sigmoid(self.conv(fused))


class CGAFusion(nn.Module):
    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn = self.pa(initial, sattn + cattn)
        mixed = initial + pattn * x + (1.0 - pattn) * y
        return self.proj(mixed)


class HistogramContrastEncoder(nn.Module):
    """Extract compact contrast-aware descriptors from luminance statistics."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        hidden = max(out_dim // 2, 8)
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], assumes normalized image/features in [0, 1] for stable stats.
        lum = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        mean = lum.mean(dim=[2, 3], keepdim=False)
        std = lum.std(dim=[2, 3], keepdim=False)
        minv = lum.amin(dim=[2, 3], keepdim=False)
        maxv = lum.amax(dim=[2, 3], keepdim=False)
        contrast = (maxv - minv) / (maxv + minv + 1e-6)
        stats = torch.cat([mean, std, minv, maxv, contrast], dim=1)
        emb = self.mlp(stats)
        return emb[:, :, None, None]


class ContrastGuidedFusion(nn.Module):
    """
    CGA++ fusion:
    original spatial/channel/pixel attention + histogram/contrast descriptor.
    """

    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.hist = HistogramContrastEncoder(dim)
        self.hist_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.out = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor, rgb_ref: torch.Tensor) -> torch.Tensor:
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        hist_prior = self.hist_proj(self.hist(rgb_ref))
        pattn = self.pa(initial, sattn + cattn + hist_prior)
        mixed = initial + pattn * x + (1.0 - pattn) * y
        return self.out(mixed)


class DEBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = default_conv(dim, dim, kernel_size, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = default_conv(dim, dim, kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = self.act(res)
        res = res + x
        res = self.conv2(res)
        return res + x


class DEABlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, reduction: int = 8) -> None:
        super().__init__()
        self.conv1 = default_conv(dim, dim, kernel_size, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = default_conv(dim, dim, kernel_size, bias=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = self.act(res)
        res = res + x
        res = self.conv2(res)
        pattn = self.pa(res, self.sa(res) + self.ca(res))
        res = res * pattn
        return res + x
