import torch
from torch import nn

from .blocks import ContrastGuidedFusion, DEABlock, DEBlock
from .edge import EdgePreservingRefinement
from .frequency import FrequencyEnhancementUnit
from .physics import AtmosphericReconstruction, PhysicsHead


class PFDEANet(nn.Module):
    """
    PF-DEA-Net:
    Physics-guided + Frequency-enhanced + Contrast-aware + Edge-preserving dehazing.
    """

    def __init__(self, base_dim: int = 32) -> None:
        super().__init__()
        # Encoder
        self.down1 = nn.Conv2d(3, base_dim, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.enc1 = nn.Sequential(DEBlock(base_dim), DEBlock(base_dim), DEBlock(base_dim))
        self.enc2 = nn.Sequential(DEBlock(base_dim * 2), DEBlock(base_dim * 2), DEBlock(base_dim * 2))
        self.enc3 = nn.Sequential(
            DEABlock(base_dim * 4),
            DEABlock(base_dim * 4),
            DEABlock(base_dim * 4),
            DEABlock(base_dim * 4),
        )

        # Physics branch
        self.physics_head = PhysicsHead(base_dim * 4)
        self.physics_layer = AtmosphericReconstruction(t0=0.05)
        self.j_phys_encoder = nn.Conv2d(3, base_dim * 4, kernel_size=3, stride=2, padding=1)

        # Frequency branch (on deep features)
        self.feu = FrequencyEnhancementUnit(base_dim * 4)

        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(DEBlock(base_dim * 2), DEBlock(base_dim * 2))
        self.dec1 = nn.Sequential(DEBlock(base_dim), DEBlock(base_dim))

        # CGA++ fusion (contrast-guided)
        self.mix3 = ContrastGuidedFusion(base_dim * 4, reduction=8)
        self.mix2 = ContrastGuidedFusion(base_dim * 2, reduction=8)
        self.mix1 = ContrastGuidedFusion(base_dim, reduction=8)

        # Edge preserving output refinement
        self.pre_out = nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1)
        self.edge_refine = EdgePreservingRefinement(channels=3)
        self.final = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, hazy: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encoder
        f1 = self.enc1(self.down1(hazy))
        f2 = self.enc2(self.down2(f1))
        f3 = self.enc3(self.down3(f2))

        # Physics branch
        t_hat, a_hat = self.physics_head(f3, target_hw=(hazy.size(2), hazy.size(3)))
        j_phys = self.physics_layer(hazy, t_hat, a_hat)
        j_phys_feat = self.j_phys_encoder(j_phys)
        if j_phys_feat.shape[-2:] != f3.shape[-2:]:
            j_phys_feat = nn.functional.interpolate(j_phys_feat, size=f3.shape[-2:], mode="bilinear", align_corners=False)

        # Frequency branch
        f3_freq = self.feu(f3)
        f3_mix = self.mix3(f3 + j_phys_feat, f3_freq, rgb_ref=hazy)

        # Decoder with CGA++ fusion
        up2 = self.up1(f3_mix)
        up2 = self.dec2(up2)
        f2_mix = self.mix2(f2, up2, rgb_ref=hazy)

        up1 = self.up2(f2_mix)
        up1 = self.dec1(up1)
        f1_mix = self.mix1(f1, up1, rgb_ref=hazy)

        # Reconstruction + edge-preserving refinement
        pre_out = torch.sigmoid(self.pre_out(f1_mix))
        refined, edge_map = self.edge_refine(pre_out, hazy)
        out = torch.sigmoid(self.final(refined))

        return {
            "dehazed": out,
            "dehazed_pre_refine": pre_out,
            "transmission": t_hat,
            "atmospheric_light": a_hat,
            "physics_reconstruction": j_phys,
            "edge_map": edge_map,
        }
