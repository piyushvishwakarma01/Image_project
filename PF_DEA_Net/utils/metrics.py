import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import piq
from scipy import ndimage
from scipy.linalg import svd
from typing import Dict, List, Tuple, Union


class MetricsCalculator:
    """Comprehensive metrics calculator for image dehazing evaluation"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize LPIPS
        try:
            self.lpips_model = piq.LPIPS().to(self.device)
        except:
            print("Warning: LPIPS not available. Install with: pip install piq")
            self.lpips_model = None
    
    def calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate PSNR"""
        # Convert to numpy
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Calculate PSNR for each image in batch
        psnr_values = []
        for i in range(pred_np.shape[0]):
            pred_img = np.transpose(pred_np[i] * 255.0, (1, 2, 0)).astype(np.uint8)
            target_img = np.transpose(target_np[i] * 255.0, (1, 2, 0)).astype(np.uint8)
            
            psnr_val = psnr(target_img, pred_img, data_range=255)
            psnr_values.append(psnr_val)
        
        return np.mean(psnr_values)
    
    def calculate_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate SSIM"""
        # Convert to numpy
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Calculate SSIM for each image in batch
        ssim_values = []
        for i in range(pred_np.shape[0]):
            pred_img = np.transpose(pred_np[i], (1, 2, 0))
            target_img = np.transpose(target_np[i], (1, 2, 0))
            
            ssim_val = ssim(target_img, pred_img, multichannel=True, data_range=1.0)
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    def calculate_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate LPIPS (Learned Perceptual Image Patch Similarity)"""
        if self.lpips_model is None:
            return 0.0
        
        # Ensure tensors are in correct range [-1, 1]
        pred_norm = pred * 2.0 - 1.0
        target_norm = target * 2.0 - 1.0
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_score = self.lpips_model(pred_norm, target_norm)
        
        return lpips_score.mean().item()
    
    def calculate_niqe(self, pred: torch.Tensor) -> float:
        """Calculate NIQE (Natural Image Quality Evaluator)"""
        try:
            # Convert to numpy
            pred_np = pred.detach().cpu().numpy()
            
            # Calculate NIQE for each image in batch
            niqe_values = []
            for i in range(pred_np.shape[0]):
                pred_img = np.transpose(pred_np[i] * 255.0, (1, 2, 0)).astype(np.uint8)
                
                # Convert to grayscale for NIQE
                pred_gray = cv2.cvtColor(pred_img, cv2.COLOR_RGB2GRAY)
                
                # Calculate NIQE using piq
                niqe_score = piq.niqe(pred_gray / 255.0)
                niqe_values.append(niqe_score.item())
            
            return np.mean(niqe_values)
        except:
            print("Warning: NIQE calculation failed")
            return 0.0
    
    def calculate_mse(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate MSE"""
        mse_loss = F.mse_loss(pred, target)
        return mse_loss.item()
    
    def calculate_mae(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate MAE"""
        mae_loss = F.l1_loss(pred, target)
        return mae_loss.item()
    
    def calculate_psnr_torch(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate PSNR using PyTorch"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    
    def calculate_ssim_torch(self, pred: torch.Tensor, target: torch.Tensor, 
                           window_size: int = 11, sigma: float = 1.5) -> float:
        """Calculate SSIM using PyTorch"""
        # Convert to grayscale
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # Create Gaussian window
        window = self._create_gaussian_window(window_size, sigma).to(pred.device)
        
        # Calculate local statistics
        mu1 = F.conv2d(pred_gray, window, padding=window_size // 2)
        mu2 = F.conv2d(target_gray, window, padding=window_size // 2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred_gray * pred_gray, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(target_gray * target_gray, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(pred_gray * target_gray, window, padding=window_size // 2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    def _create_gaussian_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian window for SSIM calculation"""
        x = torch.arange(window_size, dtype=torch.float32)
        x = x - window_size // 2
        x = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        x = x / x.sum()
        window = x.unsqueeze(1) @ x.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)
    
    def calculate_edge_preservation(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate edge preservation index"""
        # Convert to numpy
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Calculate edge preservation for each image
        ep_values = []
        for i in range(pred_np.shape[0]):
            pred_img = np.transpose(pred_np[i], (1, 2, 0))
            target_img = np.transpose(target_np[i], (1, 2, 0))
            
            # Calculate gradients
            pred_grad_x = np.abs(np.diff(pred_img, axis=1))
            pred_grad_y = np.abs(np.diff(pred_img, axis=0))
            target_grad_x = np.abs(np.diff(target_img, axis=1))
            target_grad_y = np.abs(np.diff(target_img, axis=0))
            
            # Edge preservation index
            ep_x = np.sum(pred_grad_x) / (np.sum(target_grad_x) + 1e-8)
            ep_y = np.sum(pred_grad_y) / (np.sum(target_grad_y) + 1e-8)
            ep = (ep_x + ep_y) / 2
            
            ep_values.append(ep)
        
        return np.mean(ep_values)
    
    def calculate_contrast_gain(self, pred: torch.Tensor, hazy: torch.Tensor) -> float:
        """Calculate contrast gain compared to hazy image"""
        # Convert to numpy
        pred_np = pred.detach().cpu().numpy()
        hazy_np = hazy.detach().cpu().numpy()
        
        # Calculate contrast gain for each image
        cg_values = []
        for i in range(pred_np.shape[0]):
            pred_img = np.transpose(pred_np[i], (1, 2, 0))
            hazy_img = np.transpose(hazy_np[i], (1, 2, 0))
            
            # Calculate RMS contrast
            pred_contrast = np.std(pred_img)
            hazy_contrast = np.std(hazy_img)
            
            # Contrast gain
            cg = pred_contrast / (hazy_contrast + 1e-8)
            cg_values.append(cg)
        
        return np.mean(cg_values)
    
    def calculate_haze_reduction(self, pred: torch.Tensor, hazy: torch.Tensor) -> float:
        """Calculate haze reduction metric"""
        # Convert to grayscale
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        hazy_gray = 0.299 * hazy[:, 0:1] + 0.587 * hazy[:, 1:2] + 0.114 * hazy[:, 2:3]
        
        # Calculate dark channel
        pred_dark = self._calculate_dark_channel(pred_gray)
        hazy_dark = self._calculate_dark_channel(hazy_gray)
        
        # Haze reduction
        hr = (hazy_dark.mean() - pred_dark.mean()) / hazy_dark.mean()
        
        return hr.item()
    
    def _calculate_dark_channel(self, img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
        """Calculate dark channel"""
        # Min pooling
        min_pool = F.avg_pool2d(img, kernel_size=patch_size, stride=1, padding=patch_size//2)
        dark_channel = F.avg_pool2d(min_pool, kernel_size=patch_size, stride=1, padding=patch_size//2)
        
        return dark_channel
    
    def calculate_metrics(self, pred: torch.Tensor, target: torch.Tensor, 
                       hazy: torch.Tensor = None) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['psnr'] = self.calculate_psnr(pred, target)
        metrics['ssim'] = self.calculate_ssim(pred, target)
        metrics['mse'] = self.calculate_mse(pred, target)
        metrics['mae'] = self.calculate_mae(pred, target)
        
        # Perceptual metrics
        metrics['lpips'] = self.calculate_lpips(pred, target)
        metrics['niqe'] = self.calculate_niqe(pred)
        
        # Dehazing-specific metrics
        metrics['edge_preservation'] = self.calculate_edge_preservation(pred, target)
        
        if hazy is not None:
            metrics['contrast_gain'] = self.calculate_contrast_gain(pred, hazy)
            metrics['haze_reduction'] = self.calculate_haze_reduction(pred, hazy)
        
        return metrics
    
    def calculate_batch_metrics(self, pred_batch: List[torch.Tensor], 
                             target_batch: List[torch.Tensor],
                             hazy_batch: List[torch.Tensor] = None) -> Dict[str, float]:
        """Calculate metrics for a batch of images"""
        all_metrics = []
        
        for i in range(len(pred_batch)):
            pred = pred_batch[i]
            target = target_batch[i]
            hazy = hazy_batch[i] if hazy_batch else None
            
            metrics = self.calculate_metrics(pred, target, hazy)
            all_metrics.append(metrics)
        
        # Average across batch
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics


class DehazingEvaluator:
    """Specialized evaluator for dehazing methods"""
    
    def __init__(self):
        self.metrics_calc = MetricsCalculator()
    
    def evaluate_model(self, model, dataloader, device: str = 'cuda') -> Dict[str, float]:
        """Evaluate model on dataset"""
        model.eval()
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, (hazy, target) in enumerate(dataloader):
                hazy = hazy.to(device)
                target = target.to(device)
                
                # Forward pass
                output, _, _, _ = model(hazy)
                
                # Calculate metrics
                metrics = self.metrics_calc.calculate_metrics(output, target, hazy)
                all_metrics.append(metrics)
                
                print(f"Batch {batch_idx + 1}/{len(dataloader)} - PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}")
        
        # Average across all batches
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
        
        return avg_metrics
    
    def compare_methods(self, methods_results: Dict[str, Dict[str, float]]) -> None:
        """Compare different dehazing methods"""
        print("\n" + "="*80)
        print("DEHAZING METHODS COMPARISON")
        print("="*80)
        
        # Table header
        print(f"{'Method':<20} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'NIQE':<8}")
        print("-" * 60)
        
        # Print results
        for method, metrics in methods_results.items():
            psnr = metrics.get('psnr', 0)
            ssim = metrics.get('ssim', 0)
            lpips = metrics.get('lpips', 0)
            niqe = metrics.get('niqe', 0)
            
            print(f"{method:<20} {psnr:<8.2f} {ssim:<8.4f} {lpips:<8.4f} {niqe:<8.2f}")
        
        print("="*80)


def create_metrics_report(metrics: Dict[str, float], method_name: str = "PF-DEA-Net") -> str:
    """Create a formatted metrics report"""
    report = f"\n{'='*60}\n"
    report += f"EVALUATION REPORT: {method_name}\n"
    report += f"{'='*60}\n"
    
    # Basic metrics
    report += f"Basic Metrics:\n"
    report += f"  PSNR: {metrics.get('psnr', 0):.2f} dB\n"
    report += f"  SSIM: {metrics.get('ssim', 0):.4f}\n"
    report += f"  MSE:  {metrics.get('mse', 0):.6f}\n"
    report += f"  MAE:  {metrics.get('mae', 0):.6f}\n"
    
    # Perceptual metrics
    report += f"\nPerceptual Metrics:\n"
    report += f"  LPIPS: {metrics.get('lpips', 0):.4f}\n"
    report += f"  NIQE:  {metrics.get('niqe', 0):.2f}\n"
    
    # Dehazing-specific metrics
    report += f"\nDehazing-Specific Metrics:\n"
    report += f"  Edge Preservation: {metrics.get('edge_preservation', 0):.4f}\n"
    report += f"  Contrast Gain: {metrics.get('contrast_gain', 0):.4f}\n"
    report += f"  Haze Reduction: {metrics.get('haze_reduction', 0):.4f}\n"
    
    report += f"{'='*60}\n"
    
    return report


if __name__ == "__main__":
    # Test metrics calculator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample data
    batch_size = 4
    height, width = 256, 256
    
    pred = torch.rand(batch_size, 3, height, width).to(device)
    target = torch.rand(batch_size, 3, height, width).to(device)
    hazy = torch.rand(batch_size, 3, height, width).to(device)
    
    # Calculate metrics
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_metrics(pred, target, hazy)
    
    # Print results
    print("Sample Metrics Calculation:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Create report
    report = create_metrics_report(metrics, "Sample Method")
    print(report)
