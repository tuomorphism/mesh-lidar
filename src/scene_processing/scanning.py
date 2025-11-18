"""
Module for DBSCAN utilities, used for scene processing.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class DbscanConfig:
    voxel_size: float = 0.10  # meters (your downsample size)
    eps_factor: float = 2.0  # eps = eps_factor * voxel_size
    min_samples: int = 8
    leaf_size: int = 40  # KDTree leaf_size for speed/robustness
    n_jobs: int = -1  # parallel neighbors in sklearn


def dbscan_3d(points: np.ndarray, cfg: DbscanConfig = DbscanConfig()):
    x = points.copy()
    if x.size == 0:
        return (np.full(points.shape[0], -1, dtype=int),)

    eps = cfg.eps_factor * cfg.voxel_size
    clustering = DBSCAN(
        eps=eps,
        min_samples=cfg.min_samples,
        metric="euclidean",
        leaf_size=cfg.leaf_size,
        n_jobs=cfg.n_jobs,
    ).fit(x)

    labels_sub = clustering.labels_
    return labels_sub
