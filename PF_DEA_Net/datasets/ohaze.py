import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from PIL import Image
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OHAZEDataset(Dataset):
    """O-HAZE dataset for dehazing"""
    
    def __init__(self, root_dir, split='train', patch_size=256, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        
        # Dataset paths
        if split == 'train':
            self.hazy_dir = os.path.join(root_dir, 'O-HAZE', 'train', 'hazy')
            self.clear_dir = os.path.join(root_dir, 'O-HAZE', 'train', 'clear')
        elif split == 'val':
            self.hazy_dir = os.path.join(root_dir, 'O-HAZE', 'test', 'hazy')
            self.clear_dir = os.path.join(root_dir, 'O-HAZE', 'test', 'clear')
        
        # Get image pairs
        self.hazy_images = sorted([f for f in os.listdir(self.hazy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.clear_images = sorted([f for f in os.listdir(self.clear_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Create pairs (O-HAZE has one-to-one correspondence)
        self.pairs = list(zip(self.hazy_images, self.clear_images))
        
        print(f"O-HAZE {split}: {len(self.pairs)} image pairs")
        
        # Data augmentation
        if augment:
            self.transform = A.Compose([
                A.RandomCrop(height=patch_size, width=patch_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                A.GaussNoise(var_limit=(0.0, 0.01), p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.CenterCrop(height=patch_size, width=patch_size),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        hazy_name, clear_name = self.pairs[idx]
        
        # Load images
        hazy_path = os.path.join(self.hazy_dir, hazy_name)
        clear_path = os.path.join(self.clear_dir, clear_name)
        
        hazy_img = cv2.imread(hazy_path)
        clear_img = cv2.imread(clear_path)
        
        # Convert BGR to RGB
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        hazy_img = hazy_img.astype(np.float32) / 255.0
        clear_img = clear_img.astype(np.float32) / 255.0
        
        # Apply transformations
        if self.augment or self.patch_size < min(hazy_img.shape[:2]):
            transformed = self.transform(image=hazy_img, image0=clear_img)
            hazy_tensor = transformed['image']
            clear_tensor = transformed['image0']
        else:
            # For validation/test when patch_size >= image size
            h, w = hazy_img.shape[:2]
            start_h = (h - self.patch_size) // 2
            start_w = (w - self.patch_size) // 2
            
            hazy_cropped = hazy_img[start_h:start_h+self.patch_size, start_w:start_w+self.patch_size]
            clear_cropped = clear_img[start_h:start_h+self.patch_size, start_w:start_w+self.patch_size]
            
            hazy_tensor = torch.from_numpy(hazy_cropped).permute(2, 0, 1)
            clear_tensor = torch.from_numpy(clear_cropped).permute(2, 0, 1)
        
        return hazy_tensor, clear_tensor


def create_ohaze_dataloader(root_dir, split='train', batch_size=8, patch_size=256,
                          num_workers=4, augment=True):
    """Create O-HAZE dataloader"""
    dataset = OHAZEDataset(root_dir, split, patch_size, augment)
    
    shuffle = split == 'train'
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset
    root_dir = "/path/to/your/dataset"
    
    # Test O-HAZE dataset
    ohaze_dataset = OHAZEDataset(root_dir, split='val', patch_size=256, augment=False)
    print(f"O-HAZE validation dataset size: {len(ohaze_dataset)}")
    
    # Test sample
    hazy, clear = ohaze_dataset[0]
    print(f"Hazy shape: {hazy.shape}")
    print(f"Clear shape: {clear.shape}")
    print(f"Hazy range: [{hazy.min():.3f}, {hazy.max():.3f}]")
    print(f"Clear range: [{clear.min():.3f}, {clear.max():.3f}]")
    
    # Test dataloader
    ohaze_loader = create_ohaze_dataloader(root_dir, split='val', batch_size=4)
    for batch_hazy, batch_clear in ohaze_loader:
        print(f"Batch hazy shape: {batch_hazy.shape}")
        print(f"Batch clear shape: {batch_clear.shape}")
        break
