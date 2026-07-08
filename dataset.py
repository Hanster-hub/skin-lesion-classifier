"""Step 4: custom PyTorch Dataset + augmentation pipeline for HAM10000 images."""
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from config import IMAGE_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Deterministic resize + normalize, used for val/test and inference.
standard_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Clinical-domain augmentation pipeline, used for training only.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def find_image_path(img_name, img_dirs):
    for folder in img_dirs:
        candidate = os.path.join(folder, img_name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Image {img_name} could not be found in any of: {img_dirs}")


class CustomImageDataset(Dataset):
    """Loads HAM10000 images from a metadata dataframe slice.

    Args:
        df: metadata rows (must have `image_id` and `dx_numeric`).
        img_dirs: list of folders to search for `<image_id>.jpg`.
        is_train: applies the augmentation pipeline instead of the standard one.
    """

    def __init__(self, df, img_dirs, is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = f"{row['image_id']}.jpg"
        img_path = find_image_path(img_name, self.img_dirs)
        image = Image.open(img_path).convert("RGB")

        image = train_transform(image) if self.is_train else standard_transform(image)

        label = int(row["dx_numeric"])
        return image, label, row.to_dict()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
