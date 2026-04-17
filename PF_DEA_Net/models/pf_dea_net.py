import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.deconv_plus import DEConvPlus
from .modules.histogram_attention import HistogramAttention, SpatialAttention, ChannelAttention
from .modules.fourier_enhancement import FourierEnhancement, FrequencyAttention
from .modules.edge_enhancement import EdgePreservingRefinement
from .modules.transmission_estimation import PhysicsModule


class PFDEABlock(nn.Module):
    """Enhanced DEA Block with all PF-DEA-Net components"""
    def __init__(self, dim, kernel_size=3, reduction=8):
        super(PFDEABlock, self).__init__()
        
        # Enhanced DEConv
        self.deconv = DEConvPlus(dim, frequency_aware=True)
        
        # Histogram-Guided Attention
        self.hga = HistogramAttention(dim, reduction=reduction)
        
        # Frequency Attention
        self.freq_attn = FrequencyAttention(dim, reduction=reduction)
        
        # Residual connections
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=True)
        )

    def forward(self, x):
        # Residual block with DEConv
        residual1 = x
        x = self.conv1(x)
        x = self.relu(x)
        x = x + residual1
        
        # Apply DEConv
        x_deconv = self.deconv(x)
        
        # Apply attention mechanisms
        x_hga, _ = self.hga(x_deconv)
        x_freq = self.freq_attn(x_deconv)
        
        # Fusion of attention outputs
        x_fused = torch.cat([x_deconv, x_hga, x_freq], dim=1)
        x_fused = self.fusion(x_fused)
        
        # Second residual connection
        residual2 = x
        x = self.conv2(x_fused)
        x = x + residual2
        
        return x


class PFDEANet(nn.Module):
    """Physics-Guided and Frequency-Enhanced DEA-Net"""
    def __init__(self, base_dim=32, num_blocks=[4, 4, 8, 4, 4]):
        super(PFDEANet, self).__init__()
        
        # Physics module
        self.physics_module = PhysicsModule()
        
        # Fourier enhancement
        self.fourier_enhance = FourierEnhancement(
            enhance_strength=0.1, 
            cutoff_freq=0.5, 
            filter_order=2
        )
        
        # Edge preservation
        self.edge_refine = EdgePreservingRefinement(base_dim * 4)
        
        # Encoder
        self.encoder = nn.ModuleDict()
        
        # Level 1 (input level)
        self.encoder['down1'] = nn.Conv2d(3, base_dim, 3, padding=1)
        self.encoder['level1_blocks'] = nn.ModuleList([
            PFDEABlock(base_dim) for _ in range(num_blocks[0])
        ])
        
        # Level 2
        self.encoder['down2'] = nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1)
        self.encoder['level2_blocks'] = nn.ModuleList([
            PFDEABlock(base_dim * 2) for _ in range(num_blocks[1])
        ])
        
        # Level 3 (bottleneck)
        self.encoder['down3'] = nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1)
        self.encoder['level3_blocks'] = nn.ModuleList([
            PFDEABlock(base_dim * 4) for _ in range(num_blocks[2])
        ])
        
        # Decoder
        self.decoder = nn.ModuleDict()
        
        # Level 3 to 2
        self.decoder['up3'] = nn.ConvTranspose2d(
            base_dim * 4, base_dim * 2, 3, stride=2, padding=1, output_padding=1
        )
        self.decoder['level2_blocks'] = nn.ModuleList([
            PFDEABlock(base_dim * 2) for _ in range(num_blocks[3])
        ])
        
        # Level 2 to 1
        self.decoder['up2'] = nn.ConvTranspose2d(
            base_dim * 2, base_dim, 3, stride=2, padding=1, output_padding=1
        )
        self.decoder['level1_blocks'] = nn.ModuleList([
            PFDEABlock(base_dim) for _ in range(num_blocks[4])
        ])
        
        # Output
        self.decoder['output'] = nn.Conv2d(base_dim, 3, 3, padding=1)
        
        # Feature fusion modules
        self.fusion3 = nn.Sequential(
            nn.Conv2d(base_dim * 8, base_dim * 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim * 4, base_dim * 4, 1)
        )
        
        self.fusion2 = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim * 2, base_dim * 2, 1)
        )
        
        # Attention for skip connections
        self.skip_attn3 = SpatialAttention()
        self.skip_attn2 = SpatialAttention()
        self.skip_attn1 = SpatialAttention()

    def forward(self, x):
        # Physics branch
        J_phys, t_map, A_light = self.physics_module(x)
        
        # Frequency branch
        J_freq, haze_density = self.fourier_enhance(x)
        
        # Encoder - Level 1
        x1 = self.encoder['down1'](x)
        for block in self.encoder['level1_blocks']:
            x1 = block(x1)
        
        # Encoder - Level 2
        x2 = self.encoder['down2'](x1)
        for block in self.encoder['level2_blocks']:
            x2 = block(x2)
        
        # Encoder - Level 3 (Bottleneck)
        x3 = self.encoder['down3'](x2)
        for block in self.encoder['level3_blocks']:
            x3 = block(x3)
        
        # Physics and frequency feature extraction
        J_phys_features = self.encoder['down1'](J_phys)
        J_freq_features = self.encoder['down1'](J_freq)
        
        # Physics-aware fusion at bottleneck
        physics_features = self.encoder['down3'](self.encoder['down2'](J_phys_features))
        freq_features = self.encoder['down3'](self.encoder['down2'](J_freq_features))
        
        # Fuse physics and frequency features
        x3_fused = self.fusion3(torch.cat([x3, physics_features, freq_features], dim=1))
        
        # Decoder - Level 3 to 2
        x_up3 = self.decoder['up3'](x3_fused)
        
        # Skip connection with attention
        x2_attn = self.skip_attn2(x2)
        x_up3 = x_up3 + x2_attn
        
        for block in self.decoder['level2_blocks']:
            x_up3 = block(x_up3)
        
        # Fusion at level 2
        level2_features = torch.cat([x_up3, x2], dim=1)
        x_up3_fused = self.fusion2(level2_features)
        
        # Decoder - Level 2 to 1
        x_up2 = self.decoder['up2'](x_up3_fused)
        
        # Skip connection with attention
        x1_attn = self.skip_attn1(x1)
        x_up2 = x_up2 + x1_attn
        
        for block in self.decoder['level1_blocks']:
            x_up2 = block(x_up2)
        
        # Edge-preserving refinement
        x_up2_enhanced, _ = self.edge_refine(x_up2, guidance=J_phys)
        
        # Output
        output = self.decoder['output'](x_up2_enhanced)
        
        # Residual connection with physics branch
        output = output + 0.3 * J_phys
        
        # Ensure output is in valid range
        output = torch.clamp(output, 0, 1)
        
        return output, t_map, A_light, haze_density


class PFDEANetLight(nn.Module):
    """Lightweight version of PF-DEA-Net for faster training"""
    def __init__(self, base_dim=16, num_blocks=[2, 2, 4, 2, 2]):
        super(PFDEANetLight, self).__init__()
        
        # Physics module (simplified)
        self.physics_module = PhysicsModule()
        
        # Fourier enhancement (simplified)
        self.fourier_enhance = FourierEnhancement(
            enhance_strength=0.05, 
            cutoff_freq=0.3, 
            filter_order=1
        )
        
        # Edge preservation (simplified)
        self.edge_refine = EdgePreservingRefinement(base_dim * 2)
        
        # Simplified encoder-decoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            PFDEABlock(base_dim * 2),
            nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            PFDEABlock(base_dim * 4),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            PFDEABlock(base_dim * 2),
            nn.ConvTranspose2d(base_dim * 2, base_dim, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            PFDEABlock(base_dim),
            nn.Conv2d(base_dim, 3, 3, padding=1),
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(base_dim * 8, base_dim * 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim * 4, base_dim * 4, 1)
        )

    def forward(self, x):
        # Physics branch
        J_phys, t_map, A_light = self.physics_module(x)
        
        # Frequency branch
        J_freq, haze_density = self.fourier_enhance(x)
        
        # Main network
        x_features = self.encoder(x)
        J_phys_features = self.encoder(J_phys)
        J_freq_features = self.encoder(J_freq)
        
        # Fusion
        x_fused = self.fusion(torch.cat([x_features, J_phys_features, J_freq_features], dim=1))
        
        # Decode
        output = self.decoder(x_fused)
        
        # Edge refinement
        output_enhanced, _ = self.edge_refine(output, guidance=J_phys)
        
        # Residual connection
        final_output = output_enhanced + 0.2 * J_phys
        
        return torch.clamp(final_output, 0, 1), t_map, A_light, haze_density


def create_model(model_type='full', base_dim=32):
    """Create PF-DEA-Net model"""
    if model_type == 'full':
        return PFDEANet(base_dim=base_dim)
    elif model_type == 'light':
        return PFDEANetLight(base_dim=base_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model('full', base_dim=32).to(device)
    
    # Test input
    batch_size = 2
    height, width = 256, 256
    x = torch.randn(batch_size, 3, height, width).to(device)
    
    # Forward pass
    with torch.no_grad():
        output, t_map, A_light, haze_density = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Transmission map shape: {t_map.shape}")
    print(f"Atmospheric light shape: {A_light.shape}")
    print(f"Haze density shape: {haze_density.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test lightweight model
    model_light = create_model('light', base_dim=16).to(device)
    
    with torch.no_grad():
        output_light, t_map_light, A_light_light, haze_density_light = model_light(x)
    
    print(f"\nLightweight model parameters: {sum(p.numel() for p in model_light.parameters()):,}")
    print(f"Lightweight output shape: {output_light.shape}")
