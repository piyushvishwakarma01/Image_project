import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DarkChannelPrior(nn.Module):
    """Dark Channel Prior implementation"""
    def __init__(self, patch_size=15, omega=0.95):
        super(DarkChannelPrior, self).__init__()
        self.patch_size = patch_size
        self.omega = omega
        self.padding = patch_size // 2

    def forward(self, x):
        """
        Compute dark channel
        x: input image [B, 3, H, W]
        """
        B, C, H, W = x.shape
        
        # Pad image
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        
        # Initialize dark channel
        dark_channel = torch.zeros(B, H, W, device=x.device)
        
        # Compute dark channel
        x_min, _ = torch.min(x_padded, dim=1, keepdim=True)  # Min across channels
        
        # Sliding window minimum
        kernel = torch.ones(1, 1, self.patch_size, self.patch_size, device=x.device)
        x_reshaped = x_min.view(B * 1, 1, H + 2*self.padding, W + 2*self.padding)
        dark_padded = F.conv2d(x_reshaped, kernel, padding=0)
        dark_channel = dark_padded.view(B, H, W)
        
        return dark_channel


class AtmosphericLightEstimation(nn.Module):
    """Atmospheric light estimation using dark channel prior"""
    def __init__(self, percentile=0.1):
        super(AtmosphericLightEstimation, self).__init__()
        self.percentile = percentile
        
        # CNN refinement network
        self.refine_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, dark_channel):
        """
        Estimate atmospheric light
        x: input image [B, 3, H, W]
        dark_channel: dark channel [B, H, W]
        """
        B, C, H, W = x.shape
        
        # Find brightest pixels in dark channel
        dark_flat = dark_channel.view(B, -1)
        threshold = torch.quantile(dark_flat, 1 - self.percentile/100, dim=1)
        
        # Get candidate pixels
        candidates = []
        for b in range(B):
            mask = dark_channel[b] >= threshold[b]
            candidate_pixels = x[b][:, mask]  # [3, N]
            candidates.append(candidate_pixels)
        
        # Initial atmospheric light estimation
        A_initial = torch.zeros(B, 3, device=x.device)
        for b in range(B):
            if candidates[b].shape[1] > 0:
                A_initial[b] = torch.mean(candidates[b], dim=1)
            else:
                A_initial[b] = torch.mean(x[b].view(3, -1), dim=1)
        
        # CNN refinement
        A_expanded = A_initial.view(B, 3, 1, 1).expand(-1, -1, H, W)
        A_refined = self.refine_net(x * A_expanded)
        
        # Global pooling to get final A
        A_final = F.adaptive_avg_pool2d(A_refined, 1).view(B, 3)
        
        # Ensure A is in valid range [0, 1]
        A_final = torch.clamp(A_final, 0, 1)
        
        return A_final


class TransmissionEstimation(nn.Module):
    """CNN-based transmission estimation"""
    def __init__(self, in_channels=3, base_channels=64):
        super(TransmissionEstimation, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + 1, base_channels, 3, padding=1),  # +1 for dark channel
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Skip connections
        self.skip1 = nn.Conv2d(in_channels + 1, base_channels, 1)
        self.skip2 = nn.Conv2d(base_channels, base_channels * 2, 1)

    def forward(self, x, dark_channel, A):
        """
        Estimate transmission map
        x: input image [B, 3, H, W]
        dark_channel: dark channel [B, H, W]
        A: atmospheric light [B, 3]
        """
        B, C, H, W = x.shape
        
        # Prepare input
        dark_expanded = dark_channel.unsqueeze(1)  # [B, 1, H, W]
        A_expanded = A.view(B, 3, 1, 1).expand(-1, -1, H, W)
        
        # Normalize by atmospheric light
        x_normalized = x / (A_expanded + 1e-8)
        
        # Concatenate features
        input_features = torch.cat([x_normalized, dark_expanded], dim=1)  # [B, 4, H, W]
        
        # Encoder with skip connections
        e1 = self.encoder[0](input_features)
        e1 = self.encoder[1](e1)
        skip1 = self.skip1(input_features)
        
        e2 = self.encoder[2](e1)
        e2 = self.encoder[3](e2)
        skip2 = self.skip2(e1)
        
        e3 = self.encoder[4](e2)
        e3 = self.encoder[5](e3)
        
        # Decoder with skip connections
        d1 = self.decoder[0](e3) + skip2
        d1 = self.decoder[1](d1)
        d2 = self.decoder[2](d1) + skip1
        d2 = self.decoder[3](d2)
        
        # Final transmission map
        transmission = self.decoder[4](d2)
        
        return transmission


class PhysicsBasedRecovery(nn.Module):
    """Physics-based scene recovery using atmospheric scattering model"""
    def __init__(self, t0=0.01):
        super(PhysicsBasedRecovery, self).__init__()
        self.t0 = t0
        
        # Learnable refinement
        self.refine_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, I, t, A):
        """
        Recover scene radiance using atmospheric scattering model
        I: hazy image [B, 3, H, W]
        t: transmission map [B, 1, H, W]
        A: atmospheric light [B, 3]
        """
        B, C, H, W = I.shape
        
        # Expand A to match image dimensions
        A_expanded = A.view(B, 3, 1, 1).expand(-1, -1, H, W)
        
        # Physics-based recovery
        t_clamped = torch.clamp(t, min=self.t0)
        J_phys = (I - A_expanded) / t_clamped + A_expanded
        
        # Learnable refinement
        J_refined = self.refine_net(J_phys)
        J_final = J_phys + J_refined
        
        # Ensure valid range
        J_final = torch.clamp(J_final, 0, 1)
        
        return J_final


class TransmissionRefinement(nn.Module):
    """Transmission map refinement using guided filter"""
    def __init__(self, radius=15, epsilon=0.001):
        super(TransmissionRefinement, self).__init__()
        self.radius = radius
        self.epsilon = epsilon
        
        # Learnable refinement network
        self.refine_net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),  # I + t + dark + A
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def box_filter(self, x, radius):
        """Box filter implementation"""
        kernel_size = 2 * radius + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device) / (kernel_size ** 2)
        
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        filtered = F.conv2d(x_reshaped, kernel, padding=radius)
        filtered = filtered.view(B, C, H, W)
        
        return filtered

    def guided_filter_transmission(self, I, p, radius=None, epsilon=None):
        """Guided filter for transmission refinement"""
        if radius is None:
            radius = self.radius
        if epsilon is None:
            epsilon = self.epsilon
            
        B, C, H, W = I.shape
        
        # Convert to float
        I = I.float()
        p = p.float()
        
        # Compute local means
        mean_I = self.box_filter(I, radius)
        mean_p = self.box_filter(p, radius)
        
        # Compute correlation and covariance
        mean_Ip = self.box_filter(I * p, radius)
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = self.box_filter(I * I, radius)
        var_I = mean_II - mean_I * mean_I
        
        # Compute filter coefficients
        a = cov_Ip / (var_I + epsilon)
        b = mean_p - a * mean_I
        
        # Filter coefficients
        mean_a = self.box_filter(a, radius)
        mean_b = self.box_filter(b, radius)
        
        # Apply filter
        q = mean_a * I + mean_b
        
        return q

    def forward(self, I, t_initial, dark_channel, A):
        """
        Refine transmission map
        I: input image [B, 3, H, W]
        t_initial: initial transmission [B, 1, H, W]
        dark_channel: dark channel [B, H, W]
        A: atmospheric light [B, 3]
        """
        B, C, H, W = I.shape
        
        # Guided filter refinement
        I_gray = 0.299 * I[:, 0:1] + 0.587 * I[:, 1:2] + 0.114 * I[:, 2:3]
        t_guided = self.guided_filter_transmission(I_gray, t_initial)
        
        # Prepare features for learnable refinement
        dark_expanded = dark_channel.unsqueeze(1)
        A_expanded = A.view(B, 3, 1, 1).expand(-1, -1, H, W)
        refine_features = torch.cat([I, t_guided, dark_expanded, A_expanded], dim=1)
        
        # Learnable refinement
        t_refined = self.refine_net(refine_features)
        
        # Combine guided and learned refinements
        t_final = 0.7 * t_guided + 0.3 * t_refined
        
        # Ensure valid range
        t_final = torch.clamp(t_final, 0, 1)
        
        return t_final


class PhysicsModule(nn.Module):
    """Complete physics-based dehazing module"""
    def __init__(self):
        super(PhysicsModule, self).__init__()
        
        self.dark_channel = DarkChannelPrior(patch_size=15, omega=0.95)
        self.atmospheric_light = AtmosphericLightEstimation(percentile=0.1)
        self.transmission_est = TransmissionEstimation(in_channels=3, base_channels=64)
        self.transmission_refine = TransmissionRefinement(radius=15, epsilon=0.001)
        self.physics_recovery = PhysicsBasedRecovery(t0=0.01)

    def forward(self, I):
        """
        Physics-based dehazing
        I: hazy image [B, 3, H, W]
        """
        # Dark channel prior
        dark = self.dark_channel(I)
        
        # Atmospheric light estimation
        A = self.atmospheric_light(I, dark)
        
        # Initial transmission estimation
        t_initial = self.transmission_est(I, dark, A)
        
        # Transmission refinement
        t_refined = self.transmission_refine(I, t_initial, dark, A)
        
        # Physics-based recovery
        J_phys = self.physics_recovery(I, t_refined, A)
        
        return J_phys, t_refined, A
