#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import numpy as np
from tqdm.auto import tqdm
import argparse

# ----------------------------------------
# Ovadia et al. learning rate schedule
# ----------------------------------------

def update_lr(optimizer, epoch, initial_lr):
    if epoch < 80:
        lr = initial_lr
    elif epoch < 120:
        lr = initial_lr * 0.1
    elif epoch < 160:
        lr = initial_lr * 0.01
    elif epoch < 180:
        lr = initial_lr * 0.001
    else:
        lr = initial_lr * 0.0005

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ----------------------------------------
# Build CIFAR-compatible torchvision ResNet18
# ----------------------------------------

def build_resnet18_cifar(dropout_p=0.0):
    model = resnet18(weights=None)

    # CIFAR modification: replace the first conv and remove maxpool
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    # Optional dropout: before FC only (Ovadia applies dropout everywhere,
    # but you requested "same as before" -> minimal intervention)
    if dropout_p > 0:
        model.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(512, 10)
        )
    else:
        model.fc = nn.Linear(512, 10)

    return model


# ----------------------------------------
# Train a single model
# ----------------------------------------

def train_one_model(model_id, save_dir, device="cuda",
                    initial_lr=1e-3, epochs=200, batch_size=128):
    print(f"\n=== Training model {model_id} ===")

    # Seeds for reproducibility
    torch.manual_seed(model_id)
    np.random.seed(model_id)

    # Data augmentation as in Ovadia et al.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Build model
    model = build_resnet18_cifar().to(device)

    # Adam optimizer (as required)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    criterion = nn.CrossEntropyLoss()

    # --------------------------
    # Training loop
    # --------------------------
    pbar = tqdm(range(epochs), leave=False)
    for epoch in pbar:
        model.train()
        update_lr(optimizer, epoch, initial_lr)

        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

        acc = correct / total * 100
        pbar.set_description(f"Epoch {epoch:3d} | Loss {total_loss/total:.4f} | Acc {acc:.2f}%")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_{model_id}.pth"))
    print(f"Saved model {model_id} → {save_dir}/model_{model_id}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, default=0,
                    help="ID of the training run")
    args = parser.parse_args()
    train_one_model(args.model_id, "cifar_checkpoints", device="cuda")



