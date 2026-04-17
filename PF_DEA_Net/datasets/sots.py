import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from PIL import Image
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SOTSDataset(Dataset):
    """SOTS (Synthetic Objective Testing Set) dataset for dehazing"""
    
    def __init__(self, root_dir, split='train', patch_size=256, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        
        # Dataset paths
        if split == 'train':
            self.hazy_dir = os.path.join(root_dir, 'RESIDE', 'train', 'hazy')
            self.clear_dir = os.path.join(root_dir, 'RESIDE', 'train', 'clear')
        elif split == 'val':
            self.hazy_dir = os.path.join(root_dir, 'RESIDE', 'test', 'SOTS', 'indoor', 'hazy')
            self.clear_dir = os.path.join(root_dir, 'RESIDE', 'test', 'SOTS', 'indoor', 'clear')
        elif split == 'test':
            self.hazy_dir = os.path.join(root_dir, 'RESIDE', 'test', 'SOTS', 'outdoor', 'hazy')
            self.clear_dir = os.path.join(root_dir, 'RESIDE', 'test', 'SOTS', 'outdoor', 'clear')
        
        # Get image pairs
        self.hazy_images = sorted([f for f in os.listdir(self.hazy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.clear_images = sorted([f for f in os.listdir(self.clear_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Create pairs
        self.pairs = []
        for i, hazy_img in enumerate(self.hazy_images):
            if i < len(self.clear_images):
                self.pairs.append((hazy_img, self.clear_images[i]))
        
        print(f"SOTS {split}: {len(self.pairs)} image pairs")
        
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


class ITSDataset(Dataset):
    """ITS (Indoor Training Set) dataset for dehazing"""
    
    def __init__(self, root_dir, split='train', patch_size=256, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        
        # Dataset paths
        if split == 'train':
            self.hazy_dir = os.path.join(root_dir, 'ITS', 'train', 'hazy')
            self.clear_dir = os.path.join(root_dir, 'ITS', 'train', 'clear')
        elif split == 'val':
            self.hazy_dir = os.path.join(root_dir, 'ITS', 'test', 'hazy')
            self.clear_dir = os.path.join(root_dir, 'ITS', 'test', 'clear')
        
        # Get image pairs
        self.hazy_images = sorted([f for f in os.listdir(self.hazy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.clear_images = sorted([f for f in os.listdir(self.clear_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Create pairs (ITS has multiple hazy images per clear image)
        self.pairs = []
        for hazy_img in self.hazy_images:
            # Extract base name (e.g., "1400_1" -> "1400")
            parts = hazy_img.split('_')
            if len(parts) >= 2:
                base_name = parts[0]
                clear_match = f"{base_name}.png"
                if clear_match in self.clear_images:
                    self.pairs.append((hazy_img, clear_match))
        
        print(f"ITS {split}: {len(self.pairs)} image pairs")
        
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


class OTSDataset(Dataset):
    """OTS (Outdoor Training Set) dataset for dehazing"""
    
    def __init__(self, root_dir, split='train', patch_size=256, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        
        # Dataset paths
        if split == 'train':
            self.hazy_dir = os.path.join(root_dir, 'OTS', 'train', 'hazy')
            self.clear_dir = os.path.join(root_dir, 'OTS', 'train', 'clear')
        elif split == 'val':
            self.hazy_dir = os.path.join(root_dir, 'OTS', 'test', 'hazy')
            self.clear_dir = os.path.join(root_dir, 'OTS', 'test', 'clear')
        
        # Get image pairs
        self.hazy_images = sorted([f for f in os.listdir(self.hazy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.clear_images = sorted([f for f in os.listdir(self.clear_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Create pairs
        self.pairs = []
        for hazy_img in self.hazy_images:
            # Extract base name from hazy image (remove haze parameters)
            parts = hazy_img.split('_')
            if len(parts) >= 3:
                base_name = parts[0]
                clear_match = f"{base_name}.jpg"  # OTS uses .jpg for clear images
                if clear_match in self.clear_images:
                    self.pairs.append((hazy_img, clear_match))
        
        print(f"OTS {split}: {len(self.pairs)} image pairs")
        
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


def create_sots_dataloader(root_dir, split='train', batch_size=8, patch_size=256,
                         num_workers=4, augment=True):
    """Create SOTS dataloader"""
    dataset = SOTSDataset(root_dir, split, patch_size, augment)
    
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


def create_its_dataloader(root_dir, split='train', batch_size=8, patch_size=256,
                         num_workers=4, augment=True):
    """Create ITS dataloader"""
    dataset = ITSDataset(root_dir, split, patch_size, augment)
    
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


def create_ots_dataloader(root_dir, split='train', batch_size=8, patch_size=256,
                         num_workers=4, augment=True):
    """Create OTS dataloader"""
    dataset = OTSDataset(root_dir, split, patch_size, augment)
    
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
    
    # Test SOTS dataset
    sots_dataset = SOTSDataset(root_dir, split='val', patch_size=256, augment=False)
    print(f"SOTS validation dataset size: {len(sots_dataset)}")
    
    # Test sample
    hazy, clear = sots_dataset[0]
    print(f"Hazy shape: {hazy.shape}")
    print(f"Clear shape: {clear.shape}")
    print(f"Hazy range: [{hazy.min():.3f}, {hazy.max():.3f}]")
    print(f"Clear range: [{clear.min():.3f}, {clear.max():.3f}]")
    
    # Test dataloader
    sots_loader = create_sots_dataloader(root_dir, split='val', batch_size=4)
    for batch_hazy, batch_clear in sots_loader:
        print(f"Batch hazy shape: {batch_hazy.shape}")
        print(f"Batch clear shape: {batch_clear.shape}")
        break
