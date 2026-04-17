import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class FourierEnhancement(nn.Module):
    """Fourier Transform-based Frequency Enhancement Module"""
    def __init__(self, enhance_strength=0.1, cutoff_freq=0.5, filter_order=2):
        super(FourierEnhancement, self).__init__()
        self.enhance_strength = enhance_strength
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        
        # Learnable parameters for adaptive enhancement
        self.freq_gate = nn.Parameter(torch.ones(1))
        self.cutoff_learn = nn.Parameter(torch.tensor(cutoff_freq))
        self.strength_learn = nn.Parameter(torch.tensor(enhance_strength))
        
        # Haze density estimation network
        self.haze_estimator = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def create_highpass_filter(self, height, width, cutoff_freq, order):
        """Create Butterworth high-pass filter"""
        center_h, center_w = height // 2, width // 2
        
        # Create frequency grid
        h = torch.arange(height, dtype=torch.float32, device=self.freq_gate.device)
        w = torch.arange(width, dtype=torch.float32, device=self.freq_gate.device)
        H, W = torch.meshgrid(h, w, indexing='ij')
        
        # Distance from center
        D = torch.sqrt((H - center_h) ** 2 + (W - center_w) ** 2)
        
        # Normalize distance
        D_norm = D / (min(height, width) / 2)
        
        # Butterworth high-pass filter
        H_hp = 1 / (1 + (cutoff_freq / (D_norm + 1e-8)) ** (2 * order))
        
        return H_hp

    def adaptive_frequency_enhancement(self, x, haze_density):
        """Adaptive frequency enhancement based on haze density"""
        B, C, H, W = x.shape
        
        # Apply FFT to each channel
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        # Create adaptive high-pass filter
        cutoff_freq = self.cutoff_learn * (1 + haze_density.mean())
        H_hp = self.create_highpass_filter(H, W, cutoff_freq, self.filter_order)
        
        # Apply haze-density aware enhancement
        enhance_strength = self.strength_learn * haze_density
        H_enhance = 1 + enhance_strength * H_hp.unsqueeze(0).unsqueeze(0) * self.freq_gate
        
        # Apply filter in frequency domain
        x_fft_enhanced = x_fft_shifted * H_enhance
        
        # Inverse FFT
        x_fft_enhanced_shifted = torch.fft.ifftshift(x_fft_enhanced, dim=(-2, -1))
        x_enhanced = torch.fft.ifft2(x_fft_enhanced_shifted, dim=(-2, -1)).real
        
        return x_enhanced

    def multi_scale_frequency_processing(self, x):
        """Multi-scale frequency processing"""
        scales = [1.0, 0.5, 0.25]
        enhanced_scales = []
        
        for scale in scales:
            if scale != 1.0:
                # Downsample
                H_scaled = int(x.shape[2] * scale)
                W_scaled = int(x.shape[3] * scale)
                x_scaled = F.interpolate(x, size=(H_scaled, W_scaled), mode='bilinear', align_corners=False)
            else:
                x_scaled = x
            
            # Estimate haze density for this scale
            haze_density = self.haze_estimator(x_scaled)
            
            # Apply frequency enhancement
            x_enhanced_scaled = self.adaptive_frequency_enhancement(x_scaled, haze_density)
            
            # Upsample back to original size
            if scale != 1.0:
                x_enhanced_scaled = F.interpolate(x_enhanced_scaled, size=(x.shape[2], x.shape[3]), 
                                               mode='bilinear', align_corners=False)
            
            enhanced_scales.append(x_enhanced_scaled)
        
        # Combine multi-scale results
        x_enhanced = torch.stack(enhanced_scales, dim=0).mean(dim=0)
        
        return x_enhanced

    def phase_preserving_enhancement(self, x):
        """Phase-preserving frequency enhancement"""
        B, C, H, W = x.shape
        
        # FFT to get magnitude and phase
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # Estimate haze density
        haze_density = self.haze_estimator(x)
        
        # Create frequency filter
        H_hp = self.create_highpass_filter(H, W, self.cutoff_learn, self.filter_order)
        H_enhance = 1 + self.strength_learn * haze_density * H_hp.unsqueeze(0).unsqueeze(0)
        
        # Apply enhancement to magnitude only (preserve phase)
        magnitude_enhanced = magnitude * H_enhance
        
        # Reconstruct complex signal
        x_fft_enhanced = magnitude_enhanced * torch.exp(1j * phase)
        
        # Inverse FFT
        x_enhanced = torch.fft.ifft2(x_fft_enhanced, dim=(-2, -1)).real
        
        return x_enhanced

    def forward(self, x):
        """Forward pass with frequency enhancement"""
        # Estimate haze density
        haze_density = self.haze_estimator(x)
        
        # Multi-scale frequency processing
        x_enhanced = self.multi_scale_frequency_processing(x)
        
        # Residual connection
        x_out = x + x_enhanced
        
        return x_out, haze_density


class FrequencyAttention(nn.Module):
    """Frequency domain attention mechanism"""
    def __init__(self, dim, reduction=8):
        super(FrequencyAttention, self).__init__()
        self.dim = dim
        self.reduction = reduction
        
        # Frequency analysis
        self.freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention for frequency domain
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # FFT
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        # Magnitude and phase
        magnitude = torch.abs(x_fft_shifted)
        phase = torch.angle(x_fft_shifted)
        
        # Frequency attention
        freq_attn = self.freq_conv(magnitude)
        
        # Spatial attention on frequency domain
        mag_avg = torch.mean(magnitude, dim=1, keepdim=True)
        mag_max, _ = torch.max(magnitude, dim=1, keepdim=True)
        spatial_attn_input = torch.cat([mag_avg, mag_max], dim=1)
        spatial_attn = self.spatial_attn(spatial_attn_input)
        
        # Apply attention
        magnitude_enhanced = magnitude * freq_attn * spatial_attn
        
        # Reconstruct
        x_fft_enhanced = magnitude_enhanced * torch.exp(1j * phase)
        x_fft_enhanced_shifted = torch.fft.ifftshift(x_fft_enhanced, dim=(-2, -1))
        x_enhanced = torch.fft.ifft2(x_fft_enhanced_shifted, dim=(-2, -1)).real
        
        return x_enhanced


class AdaptiveFrequencyFilter(nn.Module):
    """Adaptive frequency filter based on content"""
    def __init__(self, dim):
        super(AdaptiveFrequencyFilter, self).__init__()
        
        # Content analysis
        self.content_analyzer = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3)  # [cutoff_freq, enhance_strength, filter_order]
        )
        
        # Frequency processing parameters
        self.register_buffer('min_cutoff', torch.tensor(0.1))
        self.register_buffer('max_cutoff', torch.tensor(0.8))
        self.register_buffer('min_strength', torch.tensor(0.05))
        self.register_buffer('max_strength', torch.tensor(0.3))

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Analyze content
        params = self.content_analyzer(x)
        cutoff_freq = torch.sigmoid(params[:, 0]) * (self.max_cutoff - self.min_cutoff) + self.min_cutoff
        enhance_strength = torch.sigmoid(params[:, 1]) * (self.max_strength - self.min_strength) + self.min_strength
        filter_order = torch.relu(params[:, 2]) + 1  # Ensure positive
        
        # Apply frequency enhancement for each sample in batch
        enhanced_samples = []
        for i in range(B):
            x_i = x[i:i+1]
            
            # FFT
            x_fft = torch.fft.fft2(x_i, dim=(-2, -1))
            x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
            
            # Create filter
            h = torch.arange(H, dtype=torch.float32, device=x.device)
            w = torch.arange(W, dtype=torch.float32, device=x.device)
            H_grid, W_grid = torch.meshgrid(h, w, indexing='ij')
            
            center_h, center_w = H // 2, W // 2
            D = torch.sqrt((H_grid - center_h) ** 2 + (W_grid - center_w) ** 2)
            D_norm = D / (min(H, W) / 2)
            
            # Butterworth high-pass
            H_hp = 1 / (1 + (cutoff_freq[i] / (D_norm + 1e-8)) ** (2 * filter_order[i]))
            H_enhance = 1 + enhance_strength[i] * H_hp
            
            # Apply filter
            x_fft_enhanced = x_fft_shifted * H_enhance
            x_fft_enhanced_shifted = torch.fft.ifftshift(x_fft_enhanced, dim=(-2, -1))
            x_enhanced = torch.fft.ifft2(x_fft_enhanced_shifted, dim=(-2, -1)).real
            
            enhanced_samples.append(x_enhanced)
        
        return torch.cat(enhanced_samples, dim=0)
