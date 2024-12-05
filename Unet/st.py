import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class LungDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, img_filename)
        
        # Print statements for debugging
        print(f"Trying to load image: {img_path}")
        print(f"Trying to load mask: {mask_path}")

        try:
            # Load the image and mask
            image = Image.open(img_path).convert("L")
            mask = Image.open(mask_path).convert("L")
            
            # Apply transformations if specified
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            
            return image, mask

        except (FileNotFoundError, OSError) as e:
            # Print the error and continue to the next file
            print(f"Skipping file due to error: {e}")
            return self.__getitem__((idx + 1) % len(self))

# Example of setting up paths
image_dir = "CXR-images"
mask_dir = "Mask-images"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = LungDataset(image_dir, mask_dir, transform=transform)
for img, mask in train_dataset:
    pass  # just to iterate and trigger the print statements

images = set(os.listdir(image_dir))
masks = set(os.listdir(mask_dir))
missing_masks = images - masks

if missing_masks:
    print(f"Number of missing mask files: {len(missing_masks)}")
    print("Missing mask files for:", missing_masks)
else:
    print("All mask files are present.")

