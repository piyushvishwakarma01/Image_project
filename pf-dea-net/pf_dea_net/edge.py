import torch
from torch import nn


class LaplacianEdgeExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        w = self.kernel.repeat(c, 1, 1, 1)
        return nn.functional.conv2d(x, w, padding=1, groups=c)


class FastGuidedRefiner(nn.Module):
    """
    Lightweight guided smoothing approximation:
    learns confidence to blend raw prediction with edge-aware smoothed features.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.smooth = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, pred: torch.Tensor, guide_edges: torch.Tensor) -> torch.Tensor:
        smoothed = self.smooth(pred)
        gate = self.gate(torch.cat([pred, torch.abs(guide_edges)], dim=1))
        return gate * pred + (1.0 - gate) * smoothed


class EdgePreservingRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.edge = LaplacianEdgeExtractor()
        self.refiner = FastGuidedRefiner(channels)
        self.out = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, pred: torch.Tensor, guide_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        guide_edges = self.edge(guide_rgb)
        refined = self.refiner(pred, guide_edges)
        return self.out(refined), guide_edges
