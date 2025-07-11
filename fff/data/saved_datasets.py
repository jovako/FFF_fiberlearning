import torch
import numpy as np
import os


def get_saved_dataset(root, **kwargs):
    # check if the dataset is present in root and whether it ends with .pt or .npz
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset not found at {root}")

    try:
        train_dataset = torch.load(f"{root}/train.pt")
        val_dataset = torch.load(f"{root}/val.pt")
        test_dataset = torch.load(f"{root}/test.pt")
    except FileNotFoundError:
        # try loading numpy files
        try:
            train_dataset = np.load(f"{root}/train.npz")
            val_dataset = np.load(f"{root}/val.npz")
            test_dataset = np.load(f"{root}/test.npz")
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {root}")

        # convert numpy arrays to torch tensors
        train_dataset = [
            torch.tensor(train_dataset[key]) for key in train_dataset.keys()
        ]
        val_dataset = [torch.tensor(val_dataset[key]) for key in val_dataset.keys()]
        test_dataset = [torch.tensor(test_dataset[key]) for key in test_dataset.keys()]

        # convert to TensorDataset
        train_dataset = torch.utils.data.TensorDataset(*train_dataset)
        val_dataset = torch.utils.data.TensorDataset(*val_dataset)
        test_dataset = torch.utils.data.TensorDataset(*test_dataset)

    return train_dataset, val_dataset, test_dataset


def get_subject_model_path(root, **kwargs):
    subject_model_path = f"{root}/subject_model.pt"
    if not os.path.exists(subject_model_path):
        subject_model_path = f"{root}/subject_model/checkpoints/last.ckpt"
        if not os.path.exists(subject_model_path):
            return None
    return subject_model_path
