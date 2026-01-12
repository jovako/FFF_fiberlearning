import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from tqdm.auto import tqdm

# --------------------------------------------------
# Inception feature extractor
# --------------------------------------------------
class InceptionV3Features(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = inception_v3(
            weights="IMAGENET1K_V1",
            transform_input=False
        )
        self.model.dropout = torch.nn.Identity()
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------
# Compute mean & covariance of features
# --------------------------------------------------
def compute_statistics(images, model, batch_size=32, progress=True):
    feats = []

    iterator = range(0, len(images), batch_size)
    if progress:
        iterator = tqdm(iterator, leave=False)
    for i in iterator:
        batch = images[i:i + batch_size]

        # [-1, 1] -> [0, 1]
        batch = (batch + 1) / 2

        # Resize to Inception input
        batch = F.interpolate(batch, size=299, mode="bilinear", align_corners=False)
        with torch.no_grad():
            f = model(batch)
        feats.append(f.cpu().numpy())

    feats = np.concatenate(feats, axis=0)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


# --------------------------------------------------
# FID computation
# --------------------------------------------------
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2

    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)


# --------------------------------------------------
# Main FID function
# --------------------------------------------------
def compute_fid(images1, images2, batch_size=32, progress=True):
    """
    images1, images2: torch.Tensor of shape (N, 3, H, W), values in [-1, 1]
    """
    model = InceptionV3Features(images1.device)

    mu1, sigma1 = compute_statistics(images1, model, batch_size, progress=progress)
    mu2, sigma2 = compute_statistics(images2, model, batch_size, progress=progress)

    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return float(fid)

