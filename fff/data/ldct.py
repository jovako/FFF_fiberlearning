import os
import random
from argparse import Namespace
from typing import Dict, List

# import nibabel as nib
import numpy as np
import pydicom
import torch
from ldctinv.utils import load_yaml
from torch.utils.data import Dataset
from warnings import warn
from PIL import Image
import albumentations as A
from fff.data.utils import TrainValTest


def get_ldct_datasets(root: str, **kwargs) -> TrainValTest:
    train = LDCTMayo(mode="train", datafolder=root, **kwargs)
    val = LDCTMayo(mode="val", datafolder=root, **kwargs)
    test = LDCTMayo(mode="test", datafolder=root, **kwargs)
    return train, val, test


class LDCTMayo(Dataset):
    """Dataset class for the LDCT dataset"""

    def __init__(
        self,
        mode: str,
        datafolder: str,
        data: str = "lowdose",
        condition: str | None = None,
        seed: int = 42,
        patchsize: int = 128,
        data_subset: float = 1.0,
        data_norm: str = "meanstd",
        return_tuple: bool = True,
        resize_to: int | None = None,
        augment: bool = False,
        **kwargs,
    ):
        """Init function

        Parameters
        ----------
        mode : str
            Which subset to use. Must be `train`, `val`, or `test`.
        data: str
            Which data to use. Must be `lowdose` or `highdose`.
        condition: str
            Which condition to use. Must be `lowdose`, `highdose`, or `None`.
        datafolder : str
            Root path to datafolder
        seed : int
            Random seed to use
        patchsize : int
            Patchsize to use for training.
        data_subset : float
            Subset of the data to use.
        data_norm : str
            Normalization of the data, must be `meanstd` or `minmax`.
        return_tuple : bool
            Whether to return a tuple or a dictionary.

        Attributes
        ----------
        seed: int
            Random seed to use
        path: str
            Root path to datafolder
        patchsize: int
            Patchsize to use for training.
        data_subset: float
            Subset of the data to use.
        data_norm: str
            Normalization of the data, must be `meanstd` or `minmax`.
        return_tuple: bool
            Whether to return a tuple or a dictionary.
        info: dict
            Dictionary of `info.yml` associated with the dataset.
        samples: list
            List of all image slices
        weights: list
            List of weights associated with each slice. Slices of each patient having `n_slices` are weighted with `1/n_slices`.

        """

        # Set seeds
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.path = datafolder
        self.patchsize = patchsize
        self.data_subset = data_subset
        self.data_norm = data_norm
        self.info = load_yaml(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "ldct_info.yml")
        )
        self.return_tuple = return_tuple
        self.data = data
        self.condition = condition
        self.resize_to = resize_to
        self.use_augmentation = augment
        if self.use_augmentation:
            im_size = self.resize_to if self.resize_to else self.patchsize
            self.transforms = A.Compose(
                [
                    A.RandomResizedCrop((im_size, im_size), scale=(0.9, 1.0)),
                    A.Rotate(limit=10, interpolation=1),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.02, contrast_limit=0.02, p=0.3
                    ),
                    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.01, 0.5), p=1.0),
                ],
                additional_targets={"image2": "image"},
            )

        assert self.data in [
            "lowdose",
            "highdose",
        ], "Data must be either lowdose or highdose"
        assert self.condition in [
            None,
            "lowdose",
            "highdose",
        ], "Condition must be either lowdose or highdose"
        if self.condition == self.data:
            raise ValueError("Data and condition must be different")

        # Get all slices == samples for this split
        self.samples = [
            {**patient_dict, "slice": s + 1}
            for patient_dict in self.info[mode + "_set"]
            for s in range(patient_dict["n_slices"])
        ]
        random.shuffle(self.samples)

        self.weights = torch.tensor(
            [1.0 / patient_dict["n_slices"] for patient_dict in self.samples],
            dtype=torch.double,
        )

        if self.data_subset < 1.0:
            self.samples = self.samples[: int(len(self.samples) * self.data_subset)]
            self.weights = self.weights[: int(len(self.weights) * self.data_subset)]

    def _normalize(self, X: np.ndarray, data_norm=None) -> np.ndarray:
        """Normalize samples with precomputed mean/std or min/max

        Parameters
        ----------
        X : np.ndarray
            Array to normalize

        Returns
        -------
        np.ndarray
            Normalized array

        Raises
        ------
        ValueError
            If normalization method `self.data_norm` is neither meanstd nor minmax
        """
        if data_norm is None:
            data_norm = self.data_norm
        if data_norm == "meanstd":
            return (X - self.info["mean"]) / self.info["std"]
        elif data_norm == "minmax":
            return (X - float(self.info["min"])) / (
                float(self.info["max"]) - float(self.info["min"])
            )
        elif data_norm == "clipped":
            clipped = np.clip(
                (X - float(self.info["min"])) / float(self.info["clip_value"]),
                0,
                1,
            )
            # Scale to [-1, 1]
            return 2 * clipped - 1
        else:
            raise ValueError(f"Unknown normalization method {data_norm}")

    def denormalize(self, X: np.ndarray, data_norm=None) -> np.ndarray:
        """Denormalize samples with precomputed mean/std or min/max

        Parameters
        ----------
        X : np.ndarray
            Array to denormalize

        Returns
        -------
        np.ndarray
            Denormalized array

        Raises
        ------
        ValueError
            If normalization method `self.data_norm` is neihter meanstd nor minmax
        """
        if data_norm is None:
            data_norm = self.data_norm
        if data_norm == "meanstd":
            return X * self.info["std"] + self.info["mean"]
        elif data_norm == "minmax":
            return X * (self.info["max"] - self.info["min"]) + self.info["min"]
        elif data_norm == "clipped":
            return ((X + 1) / 2) * float(self.info["clip_value"]) + float(
                self.info["min"]
            )
        else:
            raise ValueError(f"Unknown normalization method {data_norm}")

    @staticmethod
    def _idx2filename(idx: int, n_slices: int) -> str:
        """Get filename for a patient of LDCT data given slice and number of slices in the scan.

        Parameters
        ----------
        idx : int
            Slice idx for which to return filename
        n_slices : int
            Number of slices of the scan necessary to figure out number of trailing zeros

        Returns
        -------
        str
            Filename
        """
        return "1-{}.dcm".format(str(idx).zfill(len(str(n_slices))))

    def _random_crop(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Randomly crop the same patch from images (e.g. ground truth and input)

        Parameters
        ----------
        images : List[np.ndarray]
            List of images to crop in same position for

        Returns
        -------
        List[np.ndarray]
            List of cropped images
        """
        assert all(
            [im.shape == images[0].shape for im in images]
        ), "All images must have same shape!"

        if (
            self.patchsize != images[0].shape[0] or self.patchsize != images[0].shape[1]
        ) and self.patchsize:
            x = np.random.randint(images[0].shape[0] - self.patchsize)
            y = np.random.randint(images[0].shape[1] - self.patchsize)
            images = [
                im[x : x + self.patchsize, y : y + self.patchsize] for im in images
            ]
        return images

    def _resize(self, X: np.ndarray | None) -> np.ndarray | None:
        """Resize image to `self.resize_to` using bicubic interpolation"""
        if self.resize_to is None or X is None:
            return X
        return np.array(
            Image.fromarray(X).resize((self.resize_to, self.resize_to), Image.BICUBIC)
        )

    def _augment(self, lowdose: np.ndarray | None, highdose: np.ndarray | None):
        """Apply augmentations to lowdose and highdose images"""
        if self.use_augmentation:
            data_norm = (
                "minmax" if self.data_norm in ["minmax", "meanstd"] else self.data_norm
            )
            if lowdose is not None and highdose is not None:
                lowdose, highdose = self._normalize(
                    lowdose, data_norm=data_norm
                ), self._normalize(highdose, data_norm=data_norm)
                augmented = self.transforms(image=lowdose, image2=highdose)
                lowdose = augmented["image"]
                highdose = augmented["image2"]
                lowdose, highdose = self.denormalize(
                    lowdose, data_norm=data_norm
                ), self.denormalize(highdose, data_norm=data_norm)
            elif lowdose is not None:
                lowdose = self._normalize(lowdose, data_norm=data_norm)
                lowdose = self.transforms(image=lowdose)["image"]
                lowdose = self.denormalize(lowdose, data_norm=data_norm)
            elif highdose is not None:
                highdose = self._normalize(highdose, data_norm=data_norm)
                highdose = self.transforms(image=highdose)["image"]
                highdose = self.denormalize(highdose, data_norm=data_norm)
        return lowdose, highdose

    @staticmethod
    def to_torch(X: np.ndarray) -> torch.Tensor:
        """Convert input image to torch tensor and unsqueeze"""
        return torch.unsqueeze(torch.from_numpy(X), 0)

    def reset_seed(self):
        """Reset random seeds"""
        np.random.seed(self.seed)
        random.seed(self.seed)

    def prepare_return_values(
        self, lowdose: np.ndarray | None, highdose: np.ndarray | None
    ):
        """Prepare return values"""
        data_out = lowdose if self.data == "lowdose" else highdose
        data_out = self.to_torch(self._normalize(data_out)).flatten()
        if self.condition is None:
            if self.return_tuple:
                return (data_out,)
            else:
                return {"x": data_out}

        condition_out = lowdose if self.condition == "lowdose" else highdose
        condition_out = self.to_torch(self._normalize(condition_out)).flatten()

        if self.return_tuple:
            return data_out, condition_out
        else:
            return {"x": data_out, "y": condition_out}

    def __len__(self):
        """Length of dataset"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Given an index, get slice, perform random cropping, normalize and return"""

        # Load image
        sample = self.samples[idx]
        f_name = self._idx2filename(sample["slice"], sample["n_slices"])

        if self.condition is not None:
            lowdose = pydicom.filereader.dcmread(
                os.path.join(self.path, sample["input"][2:], f_name)
            ).pixel_array.astype("float32")
            highdose = pydicom.filereader.dcmread(
                os.path.join(self.path, sample["target"][2:], f_name)
            ).pixel_array.astype("float32")

            # Crop gt and input
            lowdose, highdose = self._random_crop([lowdose, highdose])
        elif self.data == "lowdose":
            lowdose = pydicom.filereader.dcmread(
                os.path.join(self.path, sample["input"][2:], f_name)
            ).pixel_array.astype("float32")
            lowdose = self._random_crop([lowdose])[0]
            highdose = None
        elif self.data == "highdose":
            highdose = pydicom.filereader.dcmread(
                os.path.join(self.path, sample["target"][2:], f_name)
            ).pixel_array.astype("float32")
            highdose = self._random_crop([highdose])[0]
            lowdose = None
        lowdose = self._resize(lowdose)
        highdose = self._resize(highdose)

        lowdose, highdose = self._augment(lowdose, highdose)
        return self.prepare_return_values(lowdose, highdose)
