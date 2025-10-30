from typing import Tuple

import numpy as np

import torch.utils

TrainValTest = Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]


def split_dataset(data, seed=1241735):
    permuted = torch.from_numpy(np.random.default_rng(seed).permutation(data)).float()
    return (
        permuted[: int(0.8 * len(permuted))],
        permuted[int(0.8 * len(permuted)) : int(0.9 * len(permuted))],
        permuted[int(0.9 * len(permuted)) :],
    )


def normalize_ct_image(x):
    """
    Normalize a CT image to the range [0, 1].
    """
    return torch.clamp((x * 502.18507379395044 + 481.45419786099086) / 2750.0, 0, 1)


def decolorize(x_colored):
    def detect_colors(x_data):
        background_colors = torch.mean(x_data[:, :, :, 0], -1)
        return background_colors

    x_c = x_colored.reshape(-1, 3, 28, 28)
    c = detect_colors(x_c)
    # x_c = (1-x) c + x * ((c+0.5)%1)
    # --> x = (x_c-c)/((c+0.5)%1 - c)
    c = c.unsqueeze(-1).expand(-1, 3, 28 * 28).reshape(-1, 3, 28, 28)
    x_dc = (x_c - c) / ((c + 0.5) % 1 - c)
    return torch.mean(torch.abs(x_dc), 1)
