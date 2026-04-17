import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HistogramAttention(nn.Module):
    """Histogram-Guided Attention mechanism"""
    def __init__(self, dim, reduction=8, num_bins=32):
        super(HistogramAttention, self).__init__()
        self.dim = dim
        self.num_bins = num_bins
        self.reduction = reduction
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)
        
        # Channel attention
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=True),
        )
        
        # Histogram processing
        self.hist_conv = nn.Sequential(
            nn.Conv2d(num_bins, dim // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=True),
        )
        
        # Pixel attention
        self.pixel_conv = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        # Contrast computation
        self.contrast_conv = nn.Conv2d(1, dim // reduction, 3, padding=1, bias=True)
        self.contrast_mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=True),
        )

    def compute_local_histogram(self, x):
        """Compute local histogram for each pixel"""
        B, C, H, W = x.shape
        device = x.device
        
        # Normalize to [0, 1] range
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        # Create histogram bins
        hist_features = []
        for c in range(C):
            x_c = x_norm[:, c:c+1, :, :]  # [B, 1, H, W]
            
            # Local histogram computation using sliding window
            hist_c = self._sliding_histogram(x_c, self.num_bins)
            hist_features.append(hist_c)
        
        return torch.cat(hist_features, dim=1)  # [B, C*num_bins, H, W]

    def _sliding_histogram(self, x, num_bins):
        """Sliding window histogram computation"""
        B, C, H, W = x.shape
        kernel_size = 7
        padding = kernel_size // 2
        
        # Pad for sliding window
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        
        # Initialize histogram
        hist = torch.zeros(B, num_bins, H, W, device=x.device)
        
        # Compute histogram for each position
        for i in range(H):
            for j in range(W):
                # Extract local patch
                patch = x_padded[:, :, i:i+kernel_size, j:j+kernel_size]  # [B, 1, 7, 7]
                patch_flat = patch.view(B, -1)  # [B, 49]
                
                # Compute histogram
                for b in range(B):
                    hist_vals, _ = torch.histc(patch_flat[b], bins=num_bins, min=0, max=1)
                    hist[b, :, i, j] = hist_vals
        
        return hist

    def compute_local_contrast(self, x):
        """Compute local contrast using Gonzalez & Woods method"""
        # Local mean and variance
        kernel_size = 7
        padding = kernel_size // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        
        # Compute local statistics
        local_sum = F.conv2d(x_padded, torch.ones(1, 1, kernel_size, kernel_size, device=x.device), padding=0)
        local_mean = local_sum / (kernel_size ** 2)
        
        local_diff_sq = (x_padded - local_mean.unsqueeze(2).unsqueeze(3)) ** 2
        local_var_sum = F.conv2d(local_diff_sq, torch.ones(1, 1, kernel_size, kernel_size, device=x.device), padding=0)
        local_var = local_var_sum / (kernel_size ** 2)
        local_std = torch.sqrt(local_var + 1e-8)
        
        # Contrast = std / mean (Michelson contrast variant)
        contrast = local_std / (local_mean + 1e-8)
        
        return contrast

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Spatial attention
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_spatial = torch.cat([x_avg, x_max], dim=1)
        spatial_attn = self.spatial_conv(x_spatial)
        
        # Channel attention
        x_gap = F.adaptive_avg_pool2d(x, 1)
        channel_attn = self.channel_mlp(x_gap)
        
        # Histogram-guided attention
        hist_features = self.compute_local_histogram(x)  # [B, C*num_bins, H, W]
        hist_attn = self.hist_conv(hist_features)
        
        # Contrast-aware attention
        contrast = self.compute_local_contrast(x)
        contrast_attn = self.contrast_mlp(self.contrast_conv(contrast))
        
        # Combine all attentions
        combined_attn = spatial_attn + channel_attn + hist_attn + contrast_attn
        
        # Pixel attention
        x_expanded = x.unsqueeze(2)  # [B, C, 1, H, W]
        combined_expanded = combined_attn.unsqueeze(2)  # [B, C, 1, H, W]
        x_concat = torch.cat([x_expanded, combined_expanded], dim=2)  # [B, C, 2, H, W]
        x_concat = x_concat.view(B, 2 * C, H, W)
        
        pixel_attn = self.sigmoid(self.pixel_conv(x_concat))
        
        # Final attention output
        attended = x * pixel_attn
        
        return attended, pixel_attn


class SpatialAttention(nn.Module):
    """Enhanced spatial attention with multi-scale support"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, padding_mode='reflect', bias=True)
        self.conv2 = nn.Conv2d(2, 1, kernel_size*2, padding=kernel_size, padding_mode='reflect', bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Multi-scale spatial attention
        x_avg1 = torch.mean(x, dim=1, keepdim=True)
        x_max1, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([x_avg1, x_max1], dim=1)
        attn1 = self.conv1(x1)
        
        x_avg2 = F.adaptive_avg_pool2d(x, (x.size(2)//2, x.size(3)//2))
        x_max2, _ = F.adaptive_max_pool2d(x, (x.size(2)//2, x.size(3)//2))
        x2 = torch.cat([x_avg2, x_max2], dim=1)
        attn2 = self.conv2(x2)
        attn2 = F.interpolate(attn2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        # Combine multi-scale attentions
        attn = self.sigmoid(attn1 + attn2)
        
        return attn


class ChannelAttention(nn.Module):
    """Enhanced channel attention with frequency awareness"""
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gap = self.gap(x)
        gmp = self.gmp(x)
        combined = torch.cat([gap, gmp], dim=1)
        attn = self.mlp(combined)
        return self.sigmoid(attn)
