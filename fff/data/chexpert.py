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


def get_chexpert_dataset(root: str, **kwargs) -> TrainValTest:
    train_dataset = CheXpertDataset(mode="train", datafolder=root, **kwargs)
    valid_dataset = CheXpertDataset(mode="valid", datafolder=root, **kwargs)
    test_dataset = CheXpertDataset(mode="test", datafolder=root, **kwargs)
    return train_dataset, valid_dataset, test_dataset


class CheXpertDataset(Dataset):
    """
    PyTorch Dataset for the CheXpert dataset.
    Expects a CSV file (train.csv or valid.csv) with columns:
        'Path', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
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
        uncertain_policy="zeros",
        to_grayscale: bool = False,
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
            csv_file = "CheXpert-v1.0-small/train.csv"
        elif mode == "valid":
            csv_file = "CheXpert-v1.0-small/train.csv"
        elif mode == "test":
            csv_file = "CheXpert-v1.0-small/valid.csv"
        else:
            raise ValueError("mode should be 'train', 'valid' or 'test'")

        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        # only keep frontal views
        self.data = self.data[self.data["Frontal/Lateral"] == "Frontal"].reset_index(
            drop=True
        )

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

        self.uncertain_policy = uncertain_policy

        # Keep only the pathology columns
        self.pathologies = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

        # Fill missing values with 0
        self.data[self.pathologies] = self.data[self.pathologies].fillna(0)

        # Handle uncertain labels (-1)
        if self.uncertain_policy == "zeros":
            self.data[self.pathologies] = self.data[self.pathologies].replace(-1, 0)
        elif self.uncertain_policy == "ones":
            self.data[self.pathologies] = self.data[self.pathologies].replace(-1, 1)
        elif self.uncertain_policy == "ignore":
            self.data = self.data[self.data[self.pathologies].ne(-1).all(axis=1)]

        self.return_tuple = return_tuple

        transforms = []
        if patchsize is not None:
            transforms.append(A.RandomCrop((patchsize, patchsize)))

        if augment:
            transforms.extend(
                [
                    A.Rotate(limit=10, interpolation=1),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.02, contrast_limit=0.02, p=0.3
                    ),
                    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.01, 0.5), p=1.0),
                ]
            )

        if resize_to is not None:
            transforms.append(A.Resize(resize_to, resize_to))
            if augment:
                transforms.append(
                    A.RandomResizedCrop(resize_to, resize_to, scale=(0.9, 1.0))
                )
        if to_grayscale:
            transforms.append(A.ToGray(num_output_channels=1, p=1.0))
            transforms.append(
                A.Normalize(
                    mean=np.mean((0.485, 0.456, 0.406)),
                    std=np.mean((0.229, 0.224, 0.225)),
                )
            )
            self.global_mean = np.mean((0.485, 0.456, 0.406))
            self.global_std = np.mean((0.229, 0.224, 0.225))
        else:
            transforms.append(
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            )
            self.global_mean = torch.tensor((0.485, 0.456, 0.406))[None, :, None, None]
            self.global_std = torch.tensor((0.229, 0.224, 0.225))[None, :, None, None]
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

    def reset_seed(self):
        """Reset random seeds"""
        np.random.seed(self.seed)
        random.seed(self.seed)

    def prepare_return_values(self, image: np.ndarray, label: torch.Tensor):
        """Prepare return values"""
        if self.return_tuple:
            return image, label
        else:
            return {
                "image": image,
                "label": label,
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("RGB")

        label = torch.tensor(
            row[self.pathologies].values.astype(float), dtype=torch.float32
        )

        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return self.prepare_return_values(image, label)
