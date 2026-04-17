import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import logging
from tqdm import tqdm
import wandb
from datetime import datetime

from models.pf_dea_net import create_model
from models.losses import MultiComponentLoss, AdaptiveLoss
from datasets import RESIDEDataset, SOTSDataset, OHAZEDataset
from utils.metrics import MetricsCalculator
from utils.logger import setup_logger


class PFDEANetTrainer:
    """Trainer class for PF-DEA-Net"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = setup_logger(config.log_dir)
        self.logger.info(f"Training on device: {self.device}")
        
        # Initialize model
        self.model = create_model(config.model_type, config.base_dim).to(self.device)
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Initialize loss function
        if config.adaptive_loss:
            self.criterion = AdaptiveLoss()
        else:
            self.criterion = MultiComponentLoss(
                alpha=config.loss_alpha,
                beta=config.loss_beta,
                gamma=config.loss_gamma,
                delta=config.loss_delta,
                epsilon=config.loss_epsilon
            )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.lr * 0.01
        )
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize datasets
        self.setup_datasets()
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project="PF-DEA-Net",
                config=config.__dict__,
                name=config.experiment_name
            )
        
        # Best metrics tracking
        self.best_psnr = 0
        self.best_ssim = 0
        
    def setup_datasets(self):
        """Setup training and validation datasets"""
        if self.config.dataset == 'RESIDE':
            self.train_dataset = RESIDEDataset(
                root_dir=self.config.data_dir,
                split='train',
                patch_size=self.config.patch_size,
                augment=True
            )
            self.val_dataset = RESIDEDataset(
                root_dir=self.config.data_dir,
                split='val',
                patch_size=self.config.patch_size,
                augment=False
            )
        elif self.config.dataset == 'SOTS':
            self.train_dataset = SOTSDataset(
                root_dir=self.config.data_dir,
                split='train',
                patch_size=self.config.patch_size,
                augment=True
            )
            self.val_dataset = SOTSDataset(
                root_dir=self.config.data_dir,
                split='val',
                patch_size=self.config.patch_size,
                augment=False
            )
        elif self.config.dataset == 'O-HAZE':
            self.train_dataset = OHAZEDataset(
                root_dir=self.config.data_dir,
                split='train',
                patch_size=self.config.patch_size,
                augment=True
            )
            self.val_dataset = OHAZEDataset(
                root_dir=self.config.data_dir,
                split='val',
                patch_size=self.config.patch_size,
                augment=False
            )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_losses = {}
        
        # Update adaptive loss epoch
        if isinstance(self.criterion, AdaptiveLoss):
            self.criterion.update_epoch(epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, (hazy, clear) in enumerate(pbar):
            hazy = hazy.to(self.device)
            clear = clear.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output, t_map, A_light, haze_density = self.model(hazy)
            
            # Compute loss
            if isinstance(self.criterion, AdaptiveLoss):
                loss = self.criterion(output, clear)
            else:
                loss, loss_dict = self.criterion(output, clear, hazy, t_map, A_light)
                
                # Accumulate losses for logging
                for key, value in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to wandb
            if self.config.use_wandb and batch_idx % 100 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]["lr"],
                    'train/batch': epoch * len(self.train_loader) + batch_idx
                })
        
        # Calculate epoch statistics
        avg_loss = epoch_loss / len(self.train_loader)
        
        if not isinstance(self.criterion, AdaptiveLoss):
            for key in epoch_losses:
                epoch_losses[key] /= len(self.train_loader)
        
        # Log epoch results
        self.logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        if not isinstance(self.criterion, AdaptiveLoss):
            self.logger.info(f"  MSE: {epoch_losses['mse']:.4f}")
            self.logger.info(f"  SSIM: {epoch_losses['ssim']:.4f}")
            self.logger.info(f"  Perceptual: {epoch_losses['perceptual']:.4f}")
            self.logger.info(f"  Edge: {epoch_losses['edge']:.4f}")
            self.logger.info(f"  Physics: {epoch_losses['physics']:.4f}")
        
        return avg_loss, epoch_losses
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'niqe': []
        }
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for hazy, clear in pbar:
                hazy = hazy.to(self.device)
                clear = clear.to(self.device)
                
                # Forward pass
                output, t_map, A_light, haze_density = self.model(hazy)
                
                # Calculate metrics
                batch_metrics = self.metrics_calculator.calculate_metrics(
                    output, clear, hazy
                )
                
                for key in val_metrics:
                    val_metrics[key].append(batch_metrics[key])
                
                # Update progress bar
                pbar.set_postfix({
                    'PSNR': f'{batch_metrics["psnr"]:.2f}',
                    'SSIM': f'{batch_metrics["ssim"]:.4f}'
                })
        
        # Calculate average metrics
        avg_metrics = {}
        for key in val_metrics:
            avg_metrics[key] = np.mean(val_metrics[key])
        
        # Log validation results
        self.logger.info(f"Validation Results - Epoch {epoch+1}:")
        self.logger.info(f"  PSNR: {avg_metrics['psnr']:.2f} dB")
        self.logger.info(f"  SSIM: {avg_metrics['ssim']:.4f}")
        self.logger.info(f"  LPIPS: {avg_metrics['lpips']:.4f}")
        self.logger.info(f"  NIQE: {avg_metrics['niqe']:.2f}")
        
        # Save best model
        if avg_metrics['psnr'] > self.best_psnr:
            self.best_psnr = avg_metrics['psnr']
            self.save_checkpoint(epoch, 'best_psnr', avg_metrics)
        
        if avg_metrics['ssim'] > self.best_ssim:
            self.best_ssim = avg_metrics['ssim']
            self.save_checkpoint(epoch, 'best_ssim', avg_metrics)
        
        # Log to wandb
        if self.config.use_wandb:
            wandb.log({
                'val/psnr': avg_metrics['psnr'],
                'val/ssim': avg_metrics['ssim'],
                'val/lpips': avg_metrics['lpips'],
                'val/niqe': avg_metrics['niqe'],
                'epoch': epoch
            })
        
        return avg_metrics
    
    def save_checkpoint(self, epoch, name, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        save_path = os.path.join(
            self.config.checkpoint_dir,
            f"{name}_epoch_{epoch+1}.pth"
        )
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved checkpoint: {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_losses = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, 'regular', val_metrics)
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'train/epoch_loss': train_loss,
                    'epoch': epoch
                })
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best PSNR: {self.best_psnr:.2f} dB")
        self.logger.info(f"Best SSIM: {self.best_ssim:.4f}")
        
        if self.config.use_wandb:
            wandb.finish()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PF-DEA-Net')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='full', choices=['full', 'light'])
    parser.add_argument('--base_dim', type=int, default=32)
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='RESIDE', choices=['RESIDE', 'SOTS', 'O-HAZE'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=256)
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Loss parameters
    parser.add_argument('--adaptive_loss', action='store_true')
    parser.add_argument('--loss_alpha', type=float, default=1.0)
    parser.add_argument('--loss_beta', type=float, default=0.1)
    parser.add_argument('--loss_gamma', type=float, default=0.01)
    parser.add_argument('--loss_delta', type=float, default=0.1)
    parser.add_argument('--loss_epsilon', type=float, default=0.01)
    
    # Saving parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_interval', type=int, default=10)
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--experiment_name', type=str, default='pf_dea_net')
    parser.add_argument('--resume', type=str, default=None)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create trainer
    trainer = PFDEANetTrainer(args)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()
