from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrackingConf:
    """
    dataclass TrackingConf is used to determine the
    """

    # lifecycle
    missed_tracks: int = 2  # kill after this many consecutive misses
    min_points: int = 5  # min points to birth a new track
    min_confirm_hits: int = 3
    tentative_prune: int = 3

    # gating (used on top of Mahalanobis)
    max_dist: float = 3.0  # gating in meters for Euclidean XY
    w_shape: float = 0.1  # weight on relative size deviation
    shape_gate: float = 1.2  # max allowed avg relative size mismatch

    dt_default: float = 0.1  # fallback Δt
    build_point_map: bool = True

    # EKF process noise
    q_pos: float = 0.5
    q_yaw: float = np.deg2rad(10.0)
    q_v: float = 2.0
    q_omega: float = np.deg2rad(20.0)

    # EKF measurement noise
    r_pos: float = 0.3
    r_yaw: float = np.deg2rad(5.0)

    # Mahalanobis^2 threshold (for 3D measurement: px, py, yaw)
    maha_thresh: float = 14.0

    # CRTV approximation
    straight_line_eps = 0.001
