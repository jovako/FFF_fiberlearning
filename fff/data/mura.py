import os
from typing import List
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import albumentations as A
from fff.data.utils import TrainValTest
import cv2

def get_mura_dataset(root: str, **kwargs) -> TrainValTest:
    train_dataset = MURADataset(mode="train", datafolder=root, **kwargs)
    valid_dataset = MURADataset(mode="valid", datafolder=root, **kwargs)
    test_dataset = MURADataset(mode="test", datafolder=root, **kwargs)
    return train_dataset, valid_dataset, test_dataset


class MURADataset(Dataset):
    """
    PyTorch Dataset for the mura dataset.
    """

    def __init__(
        self,
        mode: str,
        datafolder: str,
        seed: int = 42,
        patchsize: int = 320,
        train_ratio: float = 0.9,
        return_tuple: bool = True,
        resize_to: int | None = None,
        augment: bool = False,
        to_grayscale: bool = False,
        normalization_type = "imagenet",
        percentile_clip = True,
        invert_flipped_polarity = True,
    ):
        """
        Args:
            mode (str): 'train', 'valid' or 'test' to specify the dataset split
            datafolder (str): Path to the folder containing the CSV files and images
            transform (callable, optional): Optional transform to be applied
                on a sample.
            uncertain_policy (str): Policy to handle uncertain labels (-1).
                Options are 'zeros', 'ones', or 'ignore'.

        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # Use valid.csv as test set, split train.csv into train/valid
        self.root_dir = datafolder
        if mode == "train":
            csv_file = "train_image_paths.csv"
        elif mode == "valid":
            csv_file = "train_image_paths.csv"
        elif mode == "test":
            csv_file = "valid_image_paths.csv"
        else:
            raise ValueError("mode should be 'train', 'valid' or 'test'")

        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file), names=["Path"])
        mask = self.data["Path"].str.contains("XR_HAND", regex=False)
        self.data = self.data[mask].reset_index(drop=True)

        if mode == "train" and train_ratio < 1.0:
            self.data = self.data.sample(
                frac=train_ratio, random_state=self.seed
            ).reset_index(drop=True)
        elif mode == "valid" and train_ratio > 0.0:
            # invert indices for validation set
            train_subset = self.data.sample(
                frac=train_ratio, random_state=self.seed
            ).reset_index(drop=True)
            self.data = self.data.drop(train_subset.index).reset_index(drop=True)

        transforms = []
        if patchsize is not None:
            transforms.append(A.RandomCrop((patchsize, patchsize)))

        if to_grayscale:
            transforms.append(A.ToGray(num_output_channels=1, p=1.0))

        transforms.append(A.Lambda(image=self.to_float))
        
        if invert_flipped_polarity:
            transforms.append(
                A.Lambda(image=self.maybe_invert_xray)
            )
            
        if percentile_clip:
            transforms.append(
                A.Lambda(image=self.apply_percentile_clip)
            )
            
        if augment:
            transforms.extend(
                [
                    A.Rotate(limit=10, interpolation=1, border_mode=cv2.BORDER_REPLICATE),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.02, contrast_limit=(-0.02, 0.02), p=0.8,
                    ),
                    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.01, 0.5), p=1.0),
                    A.HorizontalFlip(),
                ]
            )

        if resize_to is not None:
            transforms.append(A.Resize(resize_to, resize_to))

        if to_grayscale:
            if normalization_type == "imagnet":
                transforms.append(
                    A.Normalize(
                        mean=np.mean((0.485, 0.456, 0.406)),
                        std=np.mean((0.229, 0.224, 0.225)),
                        max_pixel_value=1.0,
                    )
                )
                self.global_mean = np.mean((0.485, 0.456, 0.406))
                self.global_std = np.mean((0.229, 0.224, 0.225))
            elif normalization_type == "bounded":
                transforms.append(
                    A.Normalize(
                        mean=0.5,
                        std=0.5,
                        max_pixel_value=1.0,
                    )
                )
                self.global_mean = 0.5
                self.global_std = 0.5
            elif normalization_type == "data_mean":
                transforms.append(
                    A.Normalize(
                        mean=0.3682,
                        std=0.2788,
                        max_pixel_value=1.0,
                    )
                )
                self.global_mean = 0.3682
                self.global_std = 0.2788
        else:
            if normalization_type == "imagnet":
                transforms.append(
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225), 
                        max_pixel_value=1.0,
                    )
                )
                self.global_mean = torch.tensor((0.485, 0.456, 0.406))[None, :, None, None]
                self.global_std = torch.tensor((0.229, 0.224, 0.225))[None, :, None, None]
            elif normalization_type == "bounded":
                transforms.append(
                    A.Normalize(
                        mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5),
                        max_pixel_value=1.0,
                    )
                )
                self.global_mean = torch.tensor((0.5, 0.5, 0.5))[None, :, None, None]
                self.global_std = torch.tensor((0.5, 0.5, 0.5))[None, :, None, None]
            elif normalization_type == "data_mean":
                transforms.append(
                    A.Normalize(
                        mean=(0.3682, 0.3682, 0.3682),
                        std=(0.2788, 0.2788, 0.2788),
                        max_pixel_value=1.0,
                    )
                )
                self.global_mean = torch.tensor((0.3682, 0.3682, 0.3682))[None, :, None, None]
                self.global_std = torch.tensor((0.2788, 0.2788, 0.2788))[None, :, None, None]
                
        transforms.append(A.pytorch.ToTensorV2())

        self.transform = A.Compose(transforms)

    def normalize(self, img, value_range=[0, 1]):
        # Bring to 0, 1
        img = (img + value_range[0]) / (value_range[1] - value_range[0])
        img = (img - self.global_mean) / self.global_std
        return img

    def denormalize(self, img, clamp=True, value_range=[0, 1]):
        img = img * self.global_std + self.global_mean
        # Bring into value_range
        img = img * (value_range[1] - value_range[0]) + value_range[0]
        if clamp:
            img = torch.clamp(img, *value_range)
        return img

    @staticmethod
    def to_float(img, **kwargs):
        return img/255.0
        
    @staticmethod
    def apply_percentile_clip(img, low=1.0, high=99.0, eps=1e-8, **kwargs):
        img = img.astype(np.float32) / 255.0
        lo = np.percentile(img, low)
        hi = np.percentile(img, high)
        img = np.clip(img, lo, hi)
        img = (img - lo) / (hi - lo + eps)
        return img  # float32 [0,1]


    @staticmethod
    def maybe_invert_xray(img, threshold=0.05, **kwargs):
        img = img.astype(np.float32)
    
        h, w = img.shape[:2]
        border = np.concatenate([
            img[:10].ravel(),
            img[-10:].ravel(),
            img[:, :10].ravel(),
            img[:, -10:].ravel(),
        ])
        center = img[h//4:3*h//4, w//4:3*w//4].ravel()
    
        if border.mean() > center.mean() + threshold:
            img = 1.0 - img
    
        return img


    def reset_seed(self):
        """Reset random seeds"""
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("RGB")

        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image
