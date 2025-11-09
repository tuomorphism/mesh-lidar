from math import degrees
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from lidar_types import Scene, Cluster
from pose_calculations import (
    pose_distance_SE3,
    exp_SE3,
    log_SE3,
    hat_se3,
    project_to_SO3,
    rigid_inverse,
    sanitize_T,
)
from tracking_plot import plot_scene_pair


@dataclass
class TrackState:
    pose: np.ndarray  # 4x4 T_w
    twist: np.ndarray  # 4x4 se(3) hat (units: per second)
    shape_size: np.ndarray  # (3,)


@dataclass
class Track:
    entity_id: int
    state: TrackState
    timestamp: float
    missed: int = 0


@dataclass
class TrackingConf:
    missed_tracks: int = 3  # kill after this many consecutive misses
    min_points: int = 5  # min points to birth a new track
    max_dist: float = 0.8  # gating in meters (w/ m_per_rad coupling)
    m_per_rad: float = 4.0  # convert rad -> “meters” in pose metric
    w_shape: float = 0.3  # weight on relative size deviation
    shape_gate: float = 0.6  # max allowed avg relative size mismatch
    dt_default: float = 0.1  # fallback Δt


def _pose_err(Ta, Tb, m_per_rad=2.0):
    # translational & angular error, plus your coupled metric

    Ra, ta = Ta[:3, :3], Ta[:3, 3]
    Rb, tb = Tb[:3, :3], Tb[:3, 3]
    # translation
    et = float(np.linalg.norm(tb - ta))
    # rotation angle
    tr = np.clip(np.trace(Ra.T @ Rb), -1.0, 3.0)
    ang = float(np.arccos(0.5 * (tr - 1.0)))  # radians
    # coupled metric
    d = pose_distance_SE3(Ta, Tb, lambda_m_per_rad=m_per_rad)
    return et, degrees(ang), d


class Tracker:
    def __init__(self, conf: TrackingConf) -> None:
        self.conf = conf
        self.next_entity_id = 0
        self.tracks: List[Track] = []
        self.M = 1e9

    def _prediction(
        self, pose: np.ndarray, twist_hat: np.ndarray, dt: float
    ) -> np.ndarray:
        # twist_hat is per-second; scale by dt inside the exponential
        return pose @ exp_SE3(twist_hat * dt)

    def _make_state(
        self, pose: np.ndarray, twist_hat: np.ndarray, shape_size: np.ndarray
    ) -> TrackState:
        assert pose.shape == (4, 4)
        assert twist_hat.shape == (4, 4)
        assert shape_size.shape == (3,)
        return TrackState(pose=pose, twist=twist_hat, shape_size=shape_size)

    def _extract_pose(self, cluster: Cluster) -> np.ndarray:
        """
        Pose extraction from cluster
        """
        c = cluster.geometry.centroid
        R = project_to_SO3(cluster.geometry.rotation)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = c
        return T

    def _shape_rel_dev(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.maximum(1e-6, 0.5 * (np.abs(a) + np.abs(b)))
        return float(np.mean(np.abs(a - b) / denom))

    def _cost(self, T_pred, size_pred, T_det, size_det) -> float:
        d_pose = pose_distance_SE3(T_pred, T_det, self.conf.m_per_rad)
        d_shape = self._shape_rel_dev(size_pred, size_det)
        if d_pose > self.conf.max_dist or d_shape > self.conf.shape_gate:
            return self.M
        return d_pose + self.conf.w_shape * d_shape

    def check_log_exp(self, T):
        om, v = log_SE3(T)
        T2 = exp_SE3(hat_se3(om, v))
        err = np.linalg.norm(T[:3, 3] - T2[:3, 3]) + np.linalg.norm(
            T[:3, :3] - T2[:3, :3]
        )
        assert err < 1e-3, f"exp/log inconsistency: {err}"

    def _estimate_twist_from_pair(
        self, T_prev: np.ndarray, T_now: np.ndarray, dt: float
    ) -> np.ndarray:
        T_prev_sanitized = sanitize_T(T_prev)
        T_now_sanitized = sanitize_T(T_now)

        T_rel = rigid_inverse(T_prev_sanitized) @ T_now_sanitized
        omega, v = log_SE3(T_rel)
        dt_safe = max(float(dt), 1e-6)

        self.check_log_exp(T_rel)

        return hat_se3(omega / dt_safe, v / dt_safe)

    def get_predictions(self, dt: float) -> List[np.ndarray]:
        return [
            self._prediction(tr.state.pose, tr.state.twist, dt) for tr in self.tracks
        ]

    def solve_assignment(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r, c = linear_sum_assignment(C)
        return r, c

    def _build_cost_matrix(self, detections: List[Cluster], dt: float) -> np.ndarray:
        preds = self.get_predictions(dt)
        C = np.full((len(self.tracks), len(detections)), self.M, dtype=float)
        for i, (tr, T_pred) in enumerate(zip(self.tracks, preds)):
            for j, cl in enumerate(detections):
                T_det = self._extract_pose(cl)
                C[i, j] = self._cost(
                    T_pred, tr.state.shape_size, T_det, cl.geometry.sizes
                )
        return C

    def _update_matched(
        self, pairs: List[Tuple[int, int]], detections: List[Cluster], dt: float
    ):
        for i, j in pairs:
            tr = self.tracks[i]
            cl = detections[j]
            T_meas = self._extract_pose(cl)

            twist_hat = self._estimate_twist_from_pair(tr.state.pose, T_meas, dt)
            tr.state.twist = twist_hat

            tr.state.pose = T_meas
            self.check_log_exp(T_meas)
            tr.state.shape_size = cl.geometry.sizes
            tr.timestamp += dt
            tr.missed = 0

    def _handle_unmatched(self, unmatched_tracks: List[int]):
        for i in unmatched_tracks:
            self.tracks[i].missed += 1

    def _spawn_tracks(
        self, detections: List[Cluster], unmatched_dets: List[int], t0: float
    ):
        for j in unmatched_dets:
            cl = detections[j]
            npts = len(getattr(cl, "member_indices", []))
            if npts < self.conf.min_points:
                continue
            T0 = self._extract_pose(cl)
            twist0 = np.zeros((4, 4))
            st = self._make_state(T0, twist0, cl.geometry.sizes)
            self.tracks.append(
                Track(
                    entity_id=self.next_entity_id,
                    state=st,
                    timestamp=t0,
                    missed=0,
                )
            )
            self.next_entity_id += 1

    def _prune_dead(self):
        self.tracks = [tr for tr in self.tracks if tr.missed < self.conf.missed_tracks]

    def apply(self, scenes: List[Scene]):
        def _yaw_from_R(R):
            # assume Z-up, yaw about +Z. If your axes differ, adapt here.
            # Uses atan2(sin, cos) from the 2x2 top-left submatrix.
            return float(np.arctan2(R[1, 0], R[0, 0]))

        def _ang_yaw_deg(Ra, Rb):
            ya = _yaw_from_R(Ra)
            yb = _yaw_from_R(Rb)
            d = ya - yb
            # wrap to [-pi, pi]
            d = (d + np.pi) % (2 * np.pi) - np.pi
            return abs(np.degrees(d))

        if not scenes:
            return

        # init from first scene
        first = scenes[0]
        t_first = first.timestamp or 0.0
        self.tracks = []
        for cl in first.scene_clusters or []:
            T0 = self._extract_pose(cl)
            twist0 = np.zeros((4, 4))
            st = self._make_state(T0, twist0, cl.geometry.sizes)
            self.tracks.append(
                Track(
                    entity_id=self.next_entity_id,
                    state=st,
                    timestamp=t_first,
                    missed=0,
                )
            )
            self.next_entity_id += 1

        if len(scenes) == 1:
            return

        current_scene = scenes[0]
        for k, next_scene in enumerate(scenes[1:], start=1):
            if next_scene.scene_clusters is None:
                continue

            assert current_scene.timestamp is not None
            assert next_scene.timestamp is not None
            dt = next_scene.timestamp - current_scene.timestamp
            if not np.isfinite(dt) or dt <= 0:
                dt = self.conf.dt_default

            detections = next_scene.scene_clusters or []

            # build cost, solve, filter invalid (inf) pairs
            C = self._build_cost_matrix(detections, dt)
            rows, cols = self.solve_assignment(C)
            pairs = [
                (i, j) for i, j in zip(rows.tolist(), cols.tolist()) if C[i, j] < self.M
            ]

            print(
                f"[Frame {k}] #tracks={len(self.tracks)} #dets={len(detections)} #pairs={len(pairs)}"
            )

            matched_tracks = {i for i, _ in pairs}
            matched_dets = {j for _, j in pairs}
            unmatched_tracks = [
                i for i in range(len(self.tracks)) if i not in matched_tracks
            ]
            unmatched_dets = [
                j for j in range(len(detections)) if j not in matched_dets
            ]

            # update lifecycle
            self._update_matched(pairs, detections, dt)
            self._handle_unmatched(unmatched_tracks)
            self._spawn_tracks(
                detections, unmatched_dets, t0=float(next_scene.timestamp)
            )
            self._prune_dead()

            plot_scene_pair(current_scene, next_scene, pairs)

            current_scene = next_scene
