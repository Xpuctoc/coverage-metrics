import json
import math
import os
import random
import sys
import zipfile
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.special import ndtri, ndtr
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_unzipped(zip_path, out_dir):
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = out_dir / zip_path.name.replace(" (1)", "").replace(".zip", "")
    if not expected.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    if expected.exists():
        return expected
    # Fallback: return the first csv in the extracted directory.
    csvs = sorted(out_dir.glob("*.csv"))
    return csvs[0]

def get_device(prefer_cuda):
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def load_mnist_csv(
    csv_path,
    max_rows=None,
    require_labels=False,
):
    df = pd.read_csv(csv_path, nrows=max_rows)
    labels = df["label"].to_numpy(np.int64) if "label" in df.columns else None
    pixel_cols = [c for c in df.columns if c.startswith("pixel")]
    x = df[pixel_cols].to_numpy(np.float32).reshape(-1, 1, 28, 28)
    x = x / 255.0
    x = 2.0 * x - 1.0
    return torch.from_numpy(x), labels


def normalize_image_array(x):
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().float()
    else:
        t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 2 and t.shape[1] == 784:
        t = t.view(-1, 1, 28, 28)
    elif t.ndim == 3 and t.shape[-2:] == (28, 28):
        t = t.unsqueeze(1)
    elif t.ndim == 4 and t.shape[1:] == (1, 28, 28):
        pass

    mn, mx = float(t.min()), float(t.max())
    if mx > 2.0:
        t = t / 255.0
        t = 2.0 * t - 1.0
    elif mn >= 0.0:
        t = 2.0 * t - 1.0
    else:
        t = t.clamp(-1.0, 1.0)
    return t

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.gelu(self.linear1(z))
        z = torch.tanh(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class Autoencoder(nn.Module):
    def __init__(self, latent_dims=100):
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def load_autoencoder(weights_path, device, latent_dims=100):
    ae = Autoencoder(latent_dims).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    ae.load_state_dict(state)
    ae.eval()
    return ae


@torch.no_grad()
def embed_images(
    ae,
    images,
    device,
    batch_size=512,
):
    x = normalize_image_array(images)
    outs: List[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size].to(device)
        z = ae.encoder(batch).detach().cpu().numpy().astype(np.float32)
        outs.append(z)
    return np.concatenate(outs, axis=0)


def save_local_ratio_plot(details, out_path, title="Local mass-ratio distribution"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    vals = np.log(details["local_ratio"].to_numpy() + 1e-12)
    plt.hist(vals[np.isfinite(vals)], bins=60)
    plt.axvline(-math.log(2), linestyle="--")
    plt.axvline(math.log(2), linestyle="--")
    plt.xlabel("log local ratio log(Q(B)/P(B))")
    plt.ylabel("anchors")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_curve_plot(df, out_path, x="sample_size"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g = df.groupby(x)[["completeness", "uniformity"]].agg(["mean", "std"]).reset_index()
    xs = g[x].to_numpy()
    plt.figure(figsize=(7, 4))
    for metric in ["completeness", "uniformity"]:
        mean = g[(metric, "mean")].to_numpy()
        std = g[(metric, "std")].fillna(0).to_numpy()
        plt.plot(xs, mean, marker="o", label=metric)
        plt.fill_between(xs, mean - std, mean + std, alpha=0.2)
    plt.xlabel(x)
    plt.ylabel("score")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()