import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from fff.data.utils import TrainValTest
from warnings import warn
import numpy as np

def get_cue_conflict_dataset(root: str, **kwargs) -> TrainValTest:
    train_dataset = CueConflictDataset(mode="train", root=root, **kwargs)
    valid_dataset = CueConflictDataset(mode="valid", root=root, **kwargs)
    test_dataset = CueConflictDataset(mode="test", root=root, **kwargs)
    return train_dataset, valid_dataset, test_dataset


class CueConflictDataset(Dataset):
    """
    Dataset for the texture-vs-shape cue conflict stimuli.

    Assumes directory structure:
        root/
            shape_class/
                shape_texture_*.png
    """

    def __init__(self, mode, root, resize_to=None, return_tuple=True):
        if mode in ["valid", "test"]:
            warn("Currently no splits are implemented, returning full dataset every time. Use train set to ignore this warning")

        self.root = root
        self.return_tuple = return_tuple

        self.samples = []
        self.shape_classes = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )

        self.class_to_idx = {cls: i for i, cls in enumerate(self.shape_classes)}

        for shape_cls in self.shape_classes:
            shape_dir = os.path.join(root, shape_cls)
            for fname in sorted(os.listdir(shape_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(shape_dir, fname)
                    texture_cls = self._parse_texture_class(fname)
                    self.samples.append({
                        "path": path,
                        "shape": shape_cls,
                        "texture": texture_cls
                    })

        transforms = []
        if resize_to is not None:
            transforms.append(A.Resize(resize_to, resize_to))
        transforms.append(
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
        transforms.append(A.pytorch.ToTensorV2())

        self.transform = A.Compose(transforms)

    def _strip_trailing_digits(self, s):
        return re.sub(r"\d+$", "", s)

    def _parse_texture_class(self, filename, shape_cls=None):
        name = os.path.splitext(filename)[0]
        shape_part, texture_part = name.split("-", 1)

        shape_from_name = self._strip_trailing_digits(shape_part)
        texture_cls = self._strip_trailing_digits(texture_part)

        if shape_cls is not None and shape_from_name != shape_cls:
            raise ValueError(
                f"Shape mismatch: folder={shape_cls}, filename={filename}"
            )

        return texture_cls

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = np.array(Image.open(sample["path"]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]

        if not self.return_tuple:
            return {
                "image": img,
                "shape_label": self.class_to_idx[sample["shape"]],
                "texture_label": self.class_to_idx[sample["texture"]],
                "shape_name": sample["shape"],
                "texture_name": sample["texture"],
                "path": sample["path"],
            }
        else:
            return img, self.class_to_idx[sample["shape"]], self.class_to_idx[sample["texture"]]




