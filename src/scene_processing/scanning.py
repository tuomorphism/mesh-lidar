"""
Module for DBSCAN utilities, used for scene processing.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class DbscanConfig:
    eps: float = 1.5
    min_samples: int = 5
    leaf_size: int = 40  # KDTree leaf_size for speed/robustness
    n_jobs: int = -1  # parallel neighbors in sklearn


def dbscan_3d(points: np.ndarray, cfg: DbscanConfig = DbscanConfig()):
    x = points
    if x.size == 0:
        return np.full(points.shape[0], -1, dtype=int)

    clustering = DBSCAN(
        eps=cfg.eps,
        min_samples=cfg.min_samples,
        metric="euclidean",
        leaf_size=cfg.leaf_size,
        n_jobs=cfg.n_jobs,
    ).fit(x)

    labels_sub = clustering.labels_
    return labels_sub
