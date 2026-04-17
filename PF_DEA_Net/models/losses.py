import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .modules.edge_enhancement import EdgeLoss


class MultiComponentLoss(nn.Module):
    """Multi-component loss function for PF-DEA-Net"""
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01, delta=0.1, epsilon=0.01):
        super(MultiComponentLoss, self).__init__()
        self.alpha = alpha  # MSE weight
        self.beta = beta    # SSIM weight
        self.gamma = gamma  # Perceptual weight
        self.delta = delta  # Edge weight
        self.epsilon = epsilon  # Physics weight
        
        # Perceptual loss (VGG-based)
        vgg = models.vgg16(pretrained=True)
        self.vgg_features = vgg.features[:16]  # Up to conv3_3
        for param in self.vgg_features.parameters():
            param.requires_grad = False
        
        # Edge loss
        self.edge_loss = EdgeLoss()
        
        # SSIM
        self.ssim = SSIM()
        
        # Physics consistency
        self.physics_loss = PhysicsConsistencyLoss()

    def forward(self, pred, target, I_hazy, t_map, A_light):
        """
        Compute multi-component loss
        pred: predicted dehazed image [B, 3, H, W]
        target: ground truth clear image [B, 3, H, W]
        I_hazy: input hazy image [B, 3, H, W]
        t_map: transmission map [B, 1, H, W]
        A_light: atmospheric light [B, 3]
        """
        # MSE Loss
        loss_mse = F.mse_loss(pred, target)
        
        # SSIM Loss
        loss_ssim = 1 - self.ssim(pred, target)
        
        # Perceptual Loss
        loss_perceptual = self.perceptual_loss(pred, target)
        
        # Edge Loss
        loss_edge = self.edge_loss(pred, target)
        
        # Physics Consistency Loss
        loss_physics = self.physics_loss(pred, I_hazy, t_map, A_light)
        
        # Total loss
        total_loss = (self.alpha * loss_mse + 
                    self.beta * loss_ssim + 
                    self.gamma * loss_perceptual + 
                    self.delta * loss_edge + 
                    self.epsilon * loss_physics)
        
        return total_loss, {
            'mse': loss_mse,
            'ssim': loss_ssim,
            'perceptual': loss_perceptual,
            'edge': loss_edge,
            'physics': loss_physics
        }

    def perceptual_loss(self, pred, target):
        """VGG-based perceptual loss"""
        # Normalize to VGG input range
        pred_norm = self.normalize_vgg_input(pred)
        target_norm = self.normalize_vgg_input(target)
        
        # Extract features
        pred_features = self.vgg_features(pred_norm)
        target_features = self.vgg_features(target_norm)
        
        return F.mse_loss(pred_features, target_features)

    def normalize_vgg_input(self, x):
        """Normalize for VGG input"""
        # VGG expects ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


class SSIM(nn.Module):
    """Structural Similarity Index"""
    def __init__(self, window_size=11, sigma=1.5):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer('window', self.create_window(window_size, sigma))

    def create_window(self, window_size, sigma):
        """Create Gaussian window"""
        x = torch.arange(window_size, dtype=torch.float32)
        x = x - window_size // 2
        x = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        x = x / x.sum()
        window = x.unsqueeze(1) @ x.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)

    def forward(self, img1, img2):
        """Compute SSIM"""
        B, C, H, W = img1.shape
        window = self.window.to(img1.device)
        
        # Convert to grayscale for SSIM
        img1_gray = 0.299 * img1[:, 0:1] + 0.587 * img1[:, 1:2] + 0.114 * img1[:, 2:3]
        img2_gray = 0.299 * img2[:, 0:1] + 0.587 * img2[:, 1:2] + 0.114 * img2[:, 2:3]
        
        # Compute local statistics
        mu1 = F.conv2d(img1_gray, window, padding=self.window_size // 2)
        mu2 = F.conv2d(img2_gray, window, padding=self.window_size // 2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1_gray * img1_gray, window, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2_gray * img2_gray, window, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1_gray * img2_gray, window, padding=self.window_size // 2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()


class PhysicsConsistencyLoss(nn.Module):
    """Physics consistency loss for atmospheric scattering model"""
    def __init__(self):
        super(PhysicsConsistencyLoss, self).__init__()

    def forward(self, J_pred, I_hazy, t_map, A_light):
        """
        Compute physics consistency loss
        J_pred: predicted dehazed image [B, 3, H, W]
        I_hazy: input hazy image [B, 3, H, W]
        t_map: transmission map [B, 1, H, W]
        A_light: atmospheric light [B, 3]
        """
        B, C, H, W = J_pred.shape
        
        # Expand atmospheric light
        A_expanded = A_light.view(B, 3, 1, 1).expand(-1, -1, H, W)
        
        # Reconstruct hazy image using atmospheric scattering model
        I_reconstructed = J_pred * t_map + A_expanded * (1 - t_map)
        
        # Compute reconstruction loss
        physics_loss = F.mse_loss(I_reconstructed, I_hazy)
        
        return physics_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for better feature learning"""
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, feat1, feat2, label):
        """
        Compute contrastive loss
        feat1, feat2: feature vectors [B, D]
        label: similarity label (1 for similar, 0 for dissimilar) [B]
        """
        distance = F.pairwise_distance(feat1, feat2)
        loss = label * distance.pow(2) + (1 - label) * F.relu(self.margin - distance).pow(2)
        return loss.mean()


class PerceptualColorLoss(nn.Module):
    """Perceptual loss with color preservation"""
    def __init__(self):
        super(PerceptualColorLoss, self).__init__()
        
        # Color conversion matrices
        self.register_buffer('rgb_to_lab', torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ]))
        
        self.register_buffer('lab_to_rgb', torch.tensor([
            [3.240479, -1.537150, -0.498535],
            [-0.969256, 1.875991, 0.041556],
            [0.055648, -0.204043, 1.057311]
        ]))

    def rgb_to_lab(self, rgb):
        """Convert RGB to LAB color space"""
        # Normalize to [0, 1]
        rgb = torch.clamp(rgb, 0, 1)
        
        # Apply gamma correction
        rgb = torch.where(rgb > 0.04045, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
        
        # Convert to XYZ
        rgb_flat = rgb.view(-1, 3)
        xyz = torch.mm(rgb_flat, self.rgb_to_lab.t())
        xyz = xyz.view(rgb.shape[0], 3, rgb.shape[2], rgb.shape[3])
        
        # Convert to LAB
        xyz_normalized = xyz / torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device).view(1, 3, 1, 1)
        
        f = torch.where(xyz_normalized > 0.008856, 
                      torch.pow(xyz_normalized, 1/3), 
                      7.787 * xyz_normalized + 16/116)
        
        L = 116 * f[:, 1:2] - 16
        a = 500 * (f[:, 0:1] - f[:, 1:2])
        b = 200 * (f[:, 1:2] - f[:, 2:3])
        
        return torch.cat([L, a, b], dim=1)

    def forward(self, pred, target):
        """Compute perceptual color loss in LAB space"""
        # Convert to LAB
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)
        
        # Compute loss in LAB space
        loss = F.mse_loss(pred_lab, target_lab)
        
        return loss


class TotalVariationLoss(nn.Module):
    """Total variation loss for smoothness regularization"""
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        """Compute total variation loss"""
        B, C, H, W = x.shape
        
        # Compute gradients
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        
        return (tv_h + tv_w) / (B * C * H * W)


class AdaptiveLoss(nn.Module):
    """Adaptive loss that adjusts weights based on training progress"""
    def __init__(self, initial_weights={'mse': 1.0, 'ssim': 0.1, 'perceptual': 0.01}):
        super(AdaptiveLoss, self).__init__()
        self.initial_weights = initial_weights
        self.current_epoch = 0

    def update_epoch(self, epoch):
        """Update current epoch for adaptive weighting"""
        self.current_epoch = epoch

    def get_adaptive_weights(self):
        """Get adaptive weights based on training progress"""
        # Gradually increase perceptual and SSIM weights
        progress = min(self.current_epoch / 100.0, 1.0)  # Normalize to [0, 1]
        
        weights = {
            'mse': self.initial_weights['mse'] * (1 - 0.5 * progress),
            'ssim': self.initial_weights['ssim'] * (1 + 2 * progress),
            'perceptual': self.initial_weights['perceptual'] * (1 + 5 * progress)
        }
        
        return weights

    def forward(self, pred, target):
        """Compute adaptive loss"""
        weights = self.get_adaptive_weights()
        
        # MSE
        loss_mse = F.mse_loss(pred, target)
        
        # SSIM
        ssim_loss = 1 - SSIM()(pred, target)
        
        # Perceptual
        perceptual_loss = MultiComponentLoss().perceptual_loss(pred, target)
        
        # Weighted sum
        total_loss = (weights['mse'] * loss_mse + 
                     weights['ssim'] * ssim_loss + 
                     weights['perceptual'] * perceptual_loss)
        
        return total_loss
