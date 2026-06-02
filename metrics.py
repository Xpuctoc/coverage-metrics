
import json
import math
import os
import random
import sys
import zipfile
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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

@dataclass
class CoverageResult:
    completeness: float
    uniformity: float
    mean_abs_log_ratio: float
    median_abs_log_ratio: float
    covered_fraction: float
    n_real: int
    n_fake: int
    k: int
    gamma: float
    tau: float
    delta: float


def kth_nn_radii(points, k, exclude_self=True):
    n_neighbors = k + 1 if exclude_self else k
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(points)
    dist, _ = nn.kneighbors(points, return_distance=True)
    return dist[:, -1].astype(np.float32)


# count points inside balls
def radius_counts(
    centers,
    radii,
    points,
    chunk_size=256,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    counts = np.zeros(len(centers), dtype=np.int64)
    for start in range(0, len(centers), chunk_size):
        end = min(start + chunk_size, len(centers))
        d = pairwise_distances(centers[start:end], points, metric="euclidean", n_jobs=1)
        counts[start:end] = np.sum(d <= radii[start:end, None], axis=1)
    return counts


def estimate_completeness_uniformity(
    real_z,
    fake_z,
    k=20,
    gamma=1.0,
    tau=0.25,
    delta=math.log(2.0),
    eps=1e-12,
    anchor_weights=None,
    return_details=False,
):
    real_z = np.asarray(real_z, dtype=np.float32)
    fake_z = np.asarray(fake_z, dtype=np.float32)
    n, m = len(real_z), len(fake_z)
    k_eff = min(k, n - 1)
    rho = kth_nn_radii(real_z, k=k_eff, exclude_self=True)
    radii = gamma * rho

    fake_counts = radius_counts(real_z, radii, fake_z)
    real_counts = radius_counts(real_z, radii, real_z) - 1  # remove the anchor itself
    real_counts = np.maximum(real_counts, 1)

    fake_prob = fake_counts / float(m)
    real_prob = real_counts / float(n)
    local_ratio = (fake_prob + eps) / (real_prob + eps)
    abs_log_ratio = np.abs(np.log(local_ratio))
    covered = local_ratio >= tau
    uniform = covered & (abs_log_ratio <= delta)

    if anchor_weights is not None:
        w = np.asarray(anchor_weights, dtype=np.float64)
        w = w / np.sum(w)
        completeness = float(np.sum(w * covered))
        denom = float(np.sum(w * covered))
        uniformity = float(np.sum(w * uniform) / denom) if denom > 0 else float("nan")
    else:
        completeness = float(np.mean(covered))
        uniformity = float(np.mean(abs_log_ratio[covered] <= delta)) if np.any(covered) else float("nan")

    result = CoverageResult(
        completeness=completeness,
        uniformity=uniformity,
        mean_abs_log_ratio=float(np.mean(abs_log_ratio)),
        median_abs_log_ratio=float(np.median(abs_log_ratio)),
        covered_fraction=float(np.mean(covered)),
        n_real=n,
        n_fake=m,
        k=k_eff,
        gamma=float(gamma),
        tau=float(tau),
        delta=float(delta),
    )
    if not return_details:
        return result

    details = pd.DataFrame({
        "rho": rho,
        "radius": radii,
        "real_count": real_counts,
        "fake_count": fake_counts,
        "local_ratio": local_ratio,
        "abs_log_ratio": abs_log_ratio,
        "covered": covered,
        "uniform": uniform,
    })
    return result, details
