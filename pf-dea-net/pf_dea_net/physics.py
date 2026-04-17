import torch
from torch import nn


class PhysicsHead(nn.Module):
    """Estimate transmission map t(x) and atmospheric light A."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.t_head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 2, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.a_pool = nn.AdaptiveAvgPool2d(1)
        self.a_head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 4, 3, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, deep_feat: torch.Tensor, target_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.t_head(deep_feat)
        t = nn.functional.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
        a = self.a_head(self.a_pool(deep_feat))
        return t, a


class AtmosphericReconstruction(nn.Module):
    """Invert atmospheric scattering: J = (I - A(1-t)) / max(t, t0)."""

    def __init__(self, t0: float = 0.05) -> None:
        super().__init__()
        self.t0 = t0

    def forward(self, hazy: torch.Tensor, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        t_safe = torch.clamp(t, min=self.t0, max=1.0)
        a = a.expand(-1, -1, hazy.size(2), hazy.size(3))
        j = (hazy - a * (1.0 - t_safe)) / t_safe
        return torch.clamp(j, 0.0, 1.0)
