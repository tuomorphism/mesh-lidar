from dataclasses import dataclass
from typing import List, Dict
import numpy as np

from lidar_types import TrackHistory, TrackSnapshot
from lidar_types import Scene  # if your scene type is here


@dataclass
class StaticDynamicConf:
    """
    Configuration for classifying tracks as static or dynamic.
    """

    window_sec: float = 2.0  # sliding window duration

    # EKF velocity thresholds
    static_v_max: float = 0.6
    dynamic_v_min: float = 1.0

    # Bounding-box center drift thresholds (m/s)
    static_center_drift_max: float = 1.0
    dynamic_center_drift_min: float = 1.6

    # Bounding-box size deviation thresholds (relative)
    static_rel_size_dev_max: float = 0.80  # 0.70
    dynamic_rel_size_dev_min: float = 0.95

    # Overlap thresholds
    static_overlap_min: float = 0.40  # >= 40% overlap → very static
    dynamic_overlap_max: float = 0.10  # <= 10% overlap → very dynamic


def _points_in_obb(
    points_world: np.ndarray, T_w: np.ndarray, sizes: np.ndarray
) -> np.ndarray:
    """
    Check which world-space points lie inside an oriented bounding box.
    """
    T_inv = np.linalg.inv(T_w)
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    pts_box = (T_inv @ pts_h.T).T[:, :3]

    half = sizes / 2.0 + 1e-3
    return (
        (np.abs(pts_box[:, 0]) <= half[0])
        & (np.abs(pts_box[:, 1]) <= half[1])
        & (np.abs(pts_box[:, 2]) <= half[2])
    )


def _obb_overlap(
    prev_snap: TrackSnapshot,
    curr_snap: TrackSnapshot,
    prev_scene: Scene,
    curr_scene: Scene,
) -> float:

    curr_pts = curr_scene.points[curr_snap.member_indices, :3]
    if curr_pts.shape[0] == 0:
        return 0.0

    mask = _points_in_obb(curr_pts, prev_snap.T_w, prev_snap.sizes)
    return float(np.sum(mask) / curr_pts.shape[0])


def _compute_overlap(
    history: TrackHistory, scenes: List[Scene], window_snaps: List[TrackSnapshot]
) -> float:

    if len(window_snaps) < 2:
        return 0.0

    overlaps = []
    for prev, curr in zip(window_snaps[:-1], window_snaps[1:]):
        prev_scene = scenes[prev.scene_idx]
        curr_scene = scenes[curr.scene_idx]

        overlaps.append(_obb_overlap(prev, curr, prev_scene, curr_scene))

    if not overlaps:
        return 0.0

    return float(np.mean(overlaps))


def classify_static_dynamic(
    history: TrackHistory,
    conf: StaticDynamicConf,
    scenes: List[Scene],
) -> None:
    """
    New robust classifier using:
        - EKF velocity
        - OBB center drift
        - OBB size stability
        - OBB point-overlap
        - sliding window + hysteresis
    """

    snaps: List[TrackSnapshot] = history.snapshots
    if len(snaps) < 2:
        return

    # --- Sliding window selection ---
    t_last = snaps[-1].timestamp
    t_min = t_last - conf.window_sec
    window_snaps = [s for s in snaps if s.timestamp >= t_min]
    if len(window_snaps) < 2:
        window_snaps = snaps[-2:]

    times = np.array([s.timestamp for s in window_snaps])
    v_ekf = np.array([s.x_filt[3] for s in window_snaps])
    centers = np.stack([s.T_w[:3, 3] for s in window_snaps])
    sizes = np.stack([s.sizes for s in window_snaps])

    # --- EKF mean velocity ---
    v_mean = float(np.mean(np.abs(v_ekf)))
    history.mean_speed = v_mean

    # --- Center drift speed ---
    dt = float(times[-1] - times[0])
    if dt < 1e-3:
        center_drift = 0.0
    else:
        disp = float(np.linalg.norm(centers[-1] - centers[0]))
        center_drift = disp / dt

    # --- Size deviation ---
    mean_size = np.mean(sizes, axis=0)
    size_dev = np.mean(np.linalg.norm(sizes - mean_size, axis=1))
    rel_size_dev = size_dev / (np.linalg.norm(mean_size) + 1e-6)

    # --- OBB overlap ---
    overlap_mean = _compute_overlap(history, scenes, window_snaps)

    # previous state
    prev_static = history.is_static

    # --- Static conditions ---
    static_cond = (
        v_mean < conf.static_v_max
        and center_drift < conf.static_center_drift_max
        and rel_size_dev < conf.static_rel_size_dev_max
        and overlap_mean > conf.static_overlap_min
    )

    # --- Dynamic conditions ---
    dynamic_cond = (
        v_mean > conf.dynamic_v_min
        or center_drift > conf.dynamic_center_drift_min
        or rel_size_dev > conf.dynamic_rel_size_dev_min
        or overlap_mean < conf.dynamic_overlap_max
    )

    # --- Hysteresis update ---
    if static_cond:
        history.is_static = True
    elif dynamic_cond:
        history.is_static = False
    else:
        history.is_static = prev_static
