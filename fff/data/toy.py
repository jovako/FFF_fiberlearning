from _warnings import warn

import numpy as np
import torch
from scipy.stats import vonmises
from sklearn.datasets import make_moons
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset
import pandas as pd

from fff.data.manifold import ManifoldDataset
from fff.data.utils import TrainValTest

def get_split_moons(conditional: bool = False, path: str = None):
    df = pd.read_pickle(f"data/{path}")
    # read targets and conditions from dataframe
    train_data, train_targets = (
        torch.from_numpy(df["train_x"]),
        torch.from_numpy(df["train_y"]),
    )

    center = torch.mean(train_targets)
    std = torch.std(train_targets)

    train_targets = (train_targets - center) / std
    val_data = torch.from_numpy(df["val_x"])[:]
    val_targets = ((torch.from_numpy(df["val_y"]) - center) / std)[:]
    test_data = torch.from_numpy(df["test_x"])[:]
    test_targets = ((torch.from_numpy(df["test_y"]) - center) / std)[:]

    first_val_sample = val_data[0]
    matches = torch.all(train_data == first_val_sample, dim=1)
    is_in_train_set = torch.any(matches).item()
    if is_in_train_set:
        print("Warning: val datasets are corrupted!")
    first_val_sample = test_data[0]
    matches = torch.all(train_data == first_val_sample, dim=1)
    is_in_train_set = torch.any(matches).item()
    if is_in_train_set:
        print("Warning: test datasets are corrupted!")
    
    # Collect tensors for TensorDatasets
    train_data = [train_data]
    val_data = [val_data]
    test_data = [test_data]

    # Conditions
    if conditional:
        train_data.append(train_targets)
        val_data.append(val_targets)
        test_data.append(test_targets)

    return TensorDataset(
        *train_data
    ), TensorDataset(
        *val_data
    ), TensorDataset(
        *test_data
    ), (center, std)


def get_saved_MOONS_dataset():
    """Returns a 2D dataset of two moons conditioned on distance and angle"""
    # load data from 2moons_conditional_data.pkl into pandas dataframe
    df = pd.read_pickle("AE1_data")
    # read targets and conditions from dataframe
    train_data, shift, scale, train_targets = (
        torch.from_numpy(df["train_x"]),
        torch.tensor([0., 0.]),
        torch.tensor([1., 1.]),
        torch.from_numpy(df["train_y"]),
    )

    center = torch.mean(train_targets)
    std = torch.std(train_targets)

    train_targets = (train_targets - center) / std
    val_data = torch.from_numpy(df["val_x"])
    val_targets = (torch.from_numpy(df["val_y"]) - center) / std
    test_data = torch.from_numpy(df["test_x"])
    test_targets = (torch.from_numpy(df["test_y"]) - center) / std

    data = torch.cat((train_data,val_data,test_data), 0)
    targets = torch.cat((train_targets,val_targets,test_targets), 0)

    return data, targets


def make_toy_data(kind: str, N_train=60_000, N_val=1_000, N_test=5_000, random_state=12479, center=True,
                  noise: float = 0.0, **kwargs) -> TrainValTest:
    N = N_train + N_val + N_test

    conditions = []
    manifold = None
    print(kind)
    if kind == "2moons":
        data, labels = make_moons(n_samples=N, random_state=random_state)
        if kwargs.pop("conditional", False):
            conditions.append(one_hot(torch.from_numpy(labels)))
        data = torch.Tensor(data)
    elif kind == "saved_moons":
        data, labels = get_saved_MOONS_dataset()
        if kwargs.pop("conditional", False):
            conditions.append(labels)
    elif kind == "von-mises-circle":
        theta = vonmises.rvs(1, size=N, loc=np.pi / 2, random_state=random_state)
        mode_count = kwargs.pop("mode_count", 1)
        if mode_count > 1:
            offsets = np.random.default_rng(random_state).integers(0, mode_count, size=N) * np.pi * 2
            theta = (theta + offsets) / mode_count
        x1 = np.cos(theta)
        x2 = np.sin(theta)
        data = torch.from_numpy(np.stack((x1, x2), 1)).float()
        if kwargs.pop("project", False):
            from geomstats.geometry.hypersphere import Hypersphere
            manifold = Hypersphere(1)
    elif kind == "sine":
        x1 = np.random.default_rng(random_state).normal(size=N)
        x2 = np.sin(x1 * np.pi / 2)
        data = torch.from_numpy(np.stack((x1, x2), 1)).float()
    elif kind == "corner":
        u = np.random.default_rng(random_state).uniform(size=N) * 3
        x1 = torch.where(
            u < 1,
            (u - 1) * np.pi / 2,
            torch.where(
                u < 2,
                torch.sin((u - 1) * np.pi / 2),
                1
            )
        )
        x2 = torch.where(
            u < 1,
            1,
            torch.where(
                u < 2,
                torch.cos((u - 1) * np.pi / 2),
                (2 - u) * np.pi / 2
            )
        )
        data = torch.from_numpy(np.stack((x1, x2), 1))
    elif kind == "normal":
        dimension = kwargs.pop("dimension")
        data = torch.zeros(N, dimension)
    elif kind == "linear-std":
        dimension = kwargs.pop("dimension")
        data = torch.randn(N, dimension) * torch.linspace(.5, 1.5, dimension)[None]
        if center:
            raise ValueError(f"Do not use dataset {kind=!r} together with {center=!r}, use kind='normal instead.'")
    else:
        raise ValueError(f"Dataset name {kind}")
    if kwargs != {}:
        raise ValueError(f"Found excess data_set {kwargs=}")

    perm = torch.randperm(data.size(0))
    data = data[perm]
    conditions = [c[perm] for c in conditions]
    if noise > 0:
        warn("Do not use data_set.noise > 0, instead set noise hparam directly.")
        data = data + noise * torch.randn_like(data)
    if center:
        if manifold is not None:
            raise(ValueError("Fool! You set center=True for a manifold dataset!"))
        data_mean = data.mean(0, keepdim=True)
        data -= data_mean
        data_std = data.std(0, keepdim=True)
        # Do not rescale when dimension has zero std
        data /= torch.where(data_std == 0, torch.ones_like(data_std), data_std)

    datasets = (
        TensorDataset(data[:N_train], *[c[:N_train] for c in conditions]),
        TensorDataset(data[N_train:N_train + N_val], *[c[N_train:N_train + N_val] for c in conditions]),
        TensorDataset(data[N_train + N_val:], *[c[N_train + N_val:] for c in conditions])
    )
    if manifold is not None:
        datasets = [ManifoldDataset(d, manifold) for d in datasets]
    return datasets

