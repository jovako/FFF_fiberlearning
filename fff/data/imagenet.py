import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet
from warnings import warn
import albumentations as A
from fff.data.utils import TrainValTest
from albumentations.pytorch import ToTensorV2
from PIL import Image

def get_imagenet_dataset(root: str, **kwargs) -> TrainValTest:
    train_dataset = ImageNetDataset(mode="train", root=root, **kwargs)
    valid_dataset = ImageNetDataset(mode="valid", root=root, **kwargs)
    test_dataset = ImageNetDataset(mode="test", root=root, **kwargs)
    return train_dataset, valid_dataset, test_dataset

class ImageNetDataset(Dataset):
    """
    ImageNet dataset with the same interface as CueConflictDataset.
    """

    def __init__(self, mode, root, resize_to=None, normalize=True, return_tuple=True):
        if mode not in ["train", "valid", "test"]:
            raise ValueError(f"Invalid mode: {mode}")

        # torchvision uses 'val' instead of 'valid'
        split = "val" if mode in ["valid", "test"] else "train"

        self.return_tuple = return_tuple
        try:
            self.dataset = ImageNet(root=root, split=split)
        except:
            warn(f"Could not load split {split}, leaving class uninitialized")
        transforms = []
        if resize_to is not None:
            transforms.append(A.Resize(resize_to, resize_to))

        if normalize:
            transforms.append(
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            )
        transforms.append(ToTensorV2())

        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        img = np.array(img.convert("RGB"))
        img = self.transform(image=img)["image"].float()

        if not self.return_tuple:
            return {
                "image": img,
                "label": label,
                "path": self.dataset.samples[idx][0],
            }
        else:
            return img, label
