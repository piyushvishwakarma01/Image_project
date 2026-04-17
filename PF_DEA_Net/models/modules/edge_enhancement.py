import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleLaplacian(nn.Module):
    """Multi-scale Laplacian edge detection (Gonzalez & Woods)"""
    def __init__(self, scales=[1, 2, 4]):
        super(MultiScaleLaplacian, self).__init__()
        self.scales = scales
        
        # Pre-compute Laplacian kernels for different scales
        self.register_buffer('laplacian_3x3', torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('laplacian_5x5', torch.tensor([
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2, -16, 2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0]
        ], dtype=torch.float32).view(1, 1, 5, 5))
        
        # Learnable weights for scale fusion
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

    def gaussian_blur(self, x, sigma):
        """Gaussian blur for multi-scale processing"""
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(x.device)
        
        # Apply to each channel
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        blurred = F.conv2d(x_reshaped, kernel, padding=kernel_size//2)
        blurred = blurred.view(B, C, H, W)
        
        return blurred

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create Gaussian kernel"""
        center = kernel_size // 2
        kernel = torch.zeros(kernel_size, kernel_size)
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        return kernel / kernel.sum()

    def forward(self, x):
        B, C, H, W = x.shape
        edge_maps = []
        
        for scale_idx, scale in enumerate(self.scales):
            # Apply Gaussian blur for scale
            sigma = scale * 0.5
            x_blurred = self.gaussian_blur(x, sigma)
            
            # Choose appropriate Laplacian kernel
            if scale <= 1:
                laplacian_kernel = self.laplacian_3x3
            else:
                laplacian_kernel = self.laplacian_5x5
            
            # Apply Laplacian
            x_reshaped = x_blurred.view(B * C, 1, H, W)
            edge_map = F.conv2d(x_reshaped, laplacian_kernel, padding=laplacian_kernel.shape[2]//2)
            edge_map = edge_map.view(B, C, H, W)
            
            # Take absolute value
            edge_map = torch.abs(edge_map)
            edge_maps.append(edge_map)
        
        # Weighted fusion of multi-scale edges
        scale_weights_norm = F.softmax(self.scale_weights, dim=0)
        edge_fused = sum(w * edge for w, edge in zip(scale_weights_norm, edge_maps))
        
        return edge_fused, edge_maps


class GuidedFilter(nn.Module):
    """Guided filter implementation (Gonzalez & Woods concept)"""
    def __init__(self, radius=15, epsilon=0.001):
        super(GuidedFilter, self).__init__()
        self.radius = radius
        self.epsilon = epsilon
        
    def box_filter(self, x, radius):
        """Box filter for local mean computation"""
        kernel_size = 2 * radius + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device) / (kernel_size ** 2)
        
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        filtered = F.conv2d(x_reshaped, kernel, padding=radius)
        filtered = filtered.view(B, C, H, W)
        
        return filtered

    def forward(self, I, p, radius=None, epsilon=None):
        """
        Guided filter
        I: guidance image
        p: filtering input
        """
        if radius is None:
            radius = self.radius
        if epsilon is None:
            epsilon = self.epsilon
            
        B, C, H, W = I.shape
        
        # Convert to float if needed
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


class EdgePreservingRefinement(nn.Module):
    """Edge-preserving refinement module"""
    def __init__(self, dim):
        super(EdgePreservingRefinement, self).__init__()
        
        # Multi-scale Laplacian
        self.multiscale_laplacian = MultiScaleLaplacian(scales=[1, 2, 4])
        
        # Guided filter
        self.guided_filter = GuidedFilter(radius=15, epsilon=0.001)
        
        # Learnable edge enhancement
        self.edge_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Adaptive parameters
        self.edge_strength = nn.Parameter(torch.tensor(0.1))
        self.guidance_weight = nn.Parameter(torch.tensor(0.8))

    def adaptive_contrast_enhancement(self, x):
        """Adaptive contrast enhancement (CLAHE variant)"""
        B, C, H, W = x.shape
        
        # Local statistics
        kernel_size = 7
        padding = kernel_size // 2
        
        # Local mean
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        local_sum = F.conv2d(x_padded, torch.ones(1, 1, kernel_size, kernel_size, device=x.device), padding=0)
        local_mean = local_sum / (kernel_size ** 2)
        
        # Local variance
        local_diff_sq = (x_padded - local_mean.unsqueeze(2).unsqueeze(3)) ** 2
        local_var_sum = F.conv2d(local_diff_sq, torch.ones(1, 1, kernel_size, kernel_size, device=x.device), padding=0)
        local_var = local_var_sum / (kernel_size ** 2)
        local_std = torch.sqrt(local_var + 1e-8)
        
        # Adaptive contrast enhancement
        contrast_enhanced = (x - local_mean) / (local_std + 1e-8)
        contrast_enhanced = torch.tanh(contrast_enhanced) * local_std + local_mean
        
        return contrast_enhanced

    def edge_aware_reconstruction(self, x, edges):
        """Edge-aware reconstruction using edge maps"""
        # Edge-aware weighting
        edge_weights = torch.sigmoid(edges * self.edge_strength)
        
        # Apply edge enhancement
        edge_enhanced = x * (1 + edge_weights)
        
        # Learnable edge refinement
        edge_features = torch.cat([x, edges], dim=1)
        edge_refined = self.edge_conv(edge_features)
        
        # Final reconstruction
        x_enhanced = x + edge_refined * edges
        
        return x_enhanced

    def forward(self, x, guidance=None):
        """
        Forward pass
        x: input features
        guidance: optional guidance image (e.g., physics-based output)
        """
        # Multi-scale edge detection
        edges_fused, edges_multiscale = self.multiscale_laplacian(x)
        
        # Edge-preserving smoothing using guided filter
        if guidance is not None:
            x_guided = self.guided_filter(guidance, x)
            x_guided = self.guidance_weight * x_guided + (1 - self.guidance_weight) * x
        else:
            x_guided = x
        
        # Adaptive contrast enhancement
        x_contrast = self.adaptive_contrast_enhancement(x_guided)
        
        # Edge-aware reconstruction
        x_enhanced = self.edge_aware_reconstruction(x_contrast, edges_fused)
        
        return x_enhanced, edges_fused


class EdgeLoss(nn.Module):
    """Edge-aware loss function"""
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.laplacian = MultiScaleLaplacian(scales=[1, 2])

    def forward(self, pred, target):
        # Compute edges
        pred_edges, _ = self.laplacian(pred)
        target_edges, _ = self.laplacian(target)
        
        # Edge loss
        edge_loss = F.l1_loss(pred_edges, target_edges)
        
        return edge_loss


class SobelEdgeDetector(nn.Module):
    """Sobel edge detector as alternative to Laplacian"""
    def __init__(self):
        super(SobelEdgeDetector, self).__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        
        # Sobel gradients
        grad_x = F.conv2d(x_reshaped, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.sobel_y, padding=1)
        
        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_mag = grad_mag.view(B, C, H, W)
        
        return grad_mag
