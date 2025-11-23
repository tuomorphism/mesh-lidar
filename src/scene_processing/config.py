import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    voxel_size = 0.04
    eps_factor = 8.0
    min_samples = 5
    delta_t_fallback = 1.0
    merge_gap_threshold = 2.0
    clustering_scale = np.array([1.0, 1.0, 0.5])
    velocity_max_dist = 2.0
    merge_speed_thr = 0.5

    static_speed_thr = 0.3
    moving_speed_thr = 1.5
