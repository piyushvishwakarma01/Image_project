import torch
from torch import nn


class FrequencyEnhancementUnit(nn.Module):
    """
    Fourier-domain enhancement with learnable high-frequency gain.
    """

    def __init__(self, channels: int, init_gain: float = 0.2) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), init_gain))
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )

    def _radial_mask(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, w)
        radius = torch.sqrt(xx * xx + yy * yy)
        # Emphasize outer (high-frequency) region.
        return torch.clamp(radius, 0.0, 1.0)[None, None, :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = torch.fft.fft2(x, dim=(-2, -1))
        amp = torch.abs(f)
        phase = torch.angle(f)

        mask = self._radial_mask(x.size(-2), x.size(-1), x.device, x.dtype)
        amp_boost = amp * (1.0 + torch.sigmoid(self.gain) * mask)

        real = amp_boost * torch.cos(phase)
        imag = amp_boost * torch.sin(phase)
        f_enh = torch.complex(real, imag)
        x_enh = torch.fft.ifft2(f_enh, dim=(-2, -1)).real
        return self.post(x_enh)
