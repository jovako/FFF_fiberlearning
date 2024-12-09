from typing import Tuple

import numpy as np

import torch.utils

TrainValTest = Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]

def split_dataset(data, seed=1241735):
    permuted = torch.from_numpy(np.random.default_rng(seed).permutation(data)).float()
    return (
        permuted[:int(0.8 * len(permuted))],
        permuted[int(0.8 * len(permuted)):int(0.9 * len(permuted))],
        permuted[int(0.9 * len(permuted)):]
    )

def Decolorize(x_colored):
    def detect_colors(x_data):
        background_colors = x_data[:,:,0,0]
        background_colors_e = background_colors.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,28,28)
        foreground = torch.sum((x_data - background_colors_e)**2,1)
        foreground_pixel = torch.argmax(foreground.reshape(-1,28*28), -1)
        foreground_colors = x_data.reshape(-1,3,28*28)[torch.arange(x_data.shape[0]),:,foreground_pixel]
        return background_colors, foreground_colors

    x_colored = x_colored.reshape(-1,3,28,28)
    b_colors, d_colors = detect_colors(x_colored)
    # C = X * DC + (1-X) * BC -> X = (C-BC)/(DC-BC) for all three color channels
    b_colors = b_colors.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,28,28)
    d_colors = d_colors.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,28,28)
    X = (x_colored-b_colors)/(d_colors-b_colors)
    return torch.mean(X, 1)
