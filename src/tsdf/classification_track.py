from dataclasses import dataclass
from typing import List
import numpy as np

from lidar_types import TrackHistory, TrackSnapshot
from lidar_types import Scene


@dataclass
class StaticDynamicConf:
    """
    Configuration for classifying tracks as static or dynamic.
    """

    # short window for "instantaneous" behaviour
    window_sec: float = 2.0

    # long window for accumulated travel distance
    long_window_sec: float = 5.0

    # EKF velocity thresholds
    static_v_max: float = 1.4
    dynamic_v_min: float = 1.6

    # Bounding-box center drift thresholds (m/s) over a short window
    static_center_drift_max: float = 1.6
    dynamic_center_drift_min: float = 2.2

    # Overlap thresholds
    static_overlap_min: float = 0.20
    dynamic_overlap_max: float = 0.10

    # Long-horizon travel distance (meters) in the XY-plane
    static_travel_max: float = 2.4
    dynamic_travel_min: float = 3.0


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
    Robust classifier using:
        - EKF velocity (short window)
        - OBB center drift (short window)
        - OBB point-overlap (short window)
        - Long-horizon travel distance (XY)
        - Asymmetric hysteresis based on previous state
    """

    snaps: List[TrackSnapshot] = history.snapshots
    if len(snaps) < 2:
        return

    # -----------------------------
    # Short window (instantaneous-ish)
    # -----------------------------
    t_last = snaps[-1].timestamp
    t_min_short = t_last - conf.window_sec
    window_snaps = [s for s in snaps if s.timestamp >= t_min_short]
    if len(window_snaps) < 2:
        window_snaps = snaps[-2:]

    times = np.array([s.timestamp for s in window_snaps])
    v_ekf = np.array([s.x_filt[3] for s in window_snaps])
    centers = np.stack([s.T_w[:3, 3] for s in window_snaps])

    # mean EKF speed
    v_mean = float(np.mean(np.abs(v_ekf)))
    history.mean_speed = v_mean

    # center drift speed in XY
    dt = float(times[-1] - times[0])
    if dt < 1e-3:
        center_drift = 0.0
    else:
        disp_vec = centers[-1, :2] - centers[0, :2]
        disp = float(np.linalg.norm(disp_vec))
        center_drift = disp / dt

    # OBB overlap
    overlap_mean = _compute_overlap(history, scenes, window_snaps)

    # -----------------------------
    # Long window (accumulated travel)
    # -----------------------------
    t_min_long = t_last - conf.long_window_sec
    long_snaps = [s for s in snaps if s.timestamp >= t_min_long]
    if len(long_snaps) < 2:
        long_snaps = snaps[-2:]

    long_times = np.array([s.timestamp for s in long_snaps])
    long_centers_xy = np.stack([s.T_w[:2, 3] for s in long_snaps])

    dt_long = float(long_times[-1] - long_times[0])
    if dt_long < 1e-3:
        travel_dist = 0.0
    else:
        travel_vec = long_centers_xy[-1] - long_centers_xy[0]
        travel_dist = float(np.linalg.norm(travel_vec))

    # optional: store for debugging / visualization
    history.center_drift = center_drift
    history.overlap_mean = overlap_mean
    history.travel_dist = travel_dist

    prev_static = history.is_static

    # -----------------------------
    # Compose signals
    # -----------------------------
    # "Looks slow and stable right now"
    looks_static_now = (
        v_mean < conf.static_v_max
        and center_drift < conf.static_center_drift_max
        and overlap_mean > conf.static_overlap_min
        and travel_dist < conf.static_travel_max
    )

    # "Clearly moving over horizon" (long-term)
    long_motion = travel_dist > conf.dynamic_travel_min

    # "Clearly moving right now" (short-term)
    fast_now = (
        v_mean > conf.dynamic_v_min or center_drift > conf.dynamic_center_drift_min
    )

    # overlap being very small is *supporting* evidence for dynamic,
    # but shouldn't alone flip a static building
    poor_overlap = overlap_mean < conf.dynamic_overlap_max

    # -----------------------------
    # Asymmetric hysteresis
    # -----------------------------
    if prev_static:
        # Previously STATIC:
        #   - keep it static unless we see strong, consistent motion.
        #   - require combination of signals to flip to dynamic.
        flip_to_dynamic = (
            fast_now and long_motion
        ) or (  # moving now AND has moved far
            long_motion and poor_overlap
        )  # moved far and overlap is bad

        if flip_to_dynamic:
            history.is_static = False
        else:
            history.is_static = True

    else:
        # Previously DYNAMIC:
        #   - allow it to become static only if it really looks static.
        flip_to_static = looks_static_now

        if flip_to_static:
            history.is_static = True
        else:
            history.is_static = False
