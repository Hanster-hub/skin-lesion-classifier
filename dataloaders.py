"""Step 5: build train/val/test DataLoaders for a given cross-validation fold."""
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, IMAGE_DIRECTORIES, NUM_WORKERS
from dataset import CustomImageDataset, seed_worker


def split_fold(df_metadata, fold):
    """fold == -2 is always the held-out test set; the rest cycle through CV folds."""
    train_df = df_metadata[(df_metadata["fold"] != fold) & (df_metadata["fold"] != -2)].reset_index(drop=True)
    val_df = df_metadata[df_metadata["fold"] == fold].reset_index(drop=True)
    test_df = df_metadata[df_metadata["fold"] == -2].reset_index(drop=True)
    return train_df, val_df, test_df


def build_dataloaders(df_metadata, fold, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Returns (train_loader_aug, train_loader_noaug, val_loader, test_loader)."""
    train_df, val_df, test_df = split_fold(df_metadata, fold)

    train_dataset_aug = CustomImageDataset(train_df, IMAGE_DIRECTORIES, is_train=True)
    train_dataset_noaug = CustomImageDataset(train_df, IMAGE_DIRECTORIES, is_train=False)
    val_dataset = CustomImageDataset(val_df, IMAGE_DIRECTORIES, is_train=False)
    test_dataset = CustomImageDataset(test_df, IMAGE_DIRECTORIES, is_train=False)

    pin_memory = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    train_loader_aug = DataLoader(train_dataset_aug, shuffle=True, worker_init_fn=seed_worker, **common)
    train_loader_noaug = DataLoader(train_dataset_noaug, shuffle=True, worker_init_fn=seed_worker, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    return train_loader_aug, train_loader_noaug, val_loader, test_loader
