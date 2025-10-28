from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from lidar_types import Scene, Cluster
from pose_calculations import pose_distance_SE3, exp_SE3, log_SE3, hat_se3

# ----------------- Data structures -----------------


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
    last_pose: Optional[np.ndarray] = None
    missed: int = 0


@dataclass
class TrackingConf:
    missed_tracks: int = 5  # kill after this many consecutive misses
    min_points: int = 5  # min points to birth a new track
    max_dist: float = 3.0  # gating in meters (w/ m_per_rad coupling)
    m_per_rad: float = 2.0  # convert rad -> “meters” in pose metric
    w_shape: float = 0.3  # weight on relative size deviation
    shape_gate: float = 0.6  # max allowed avg relative size mismatch
    dt_default: float = 1.0  # fallback Δt


# ----------------- Tracker -----------------


class Tracker:
    def __init__(self, conf: TrackingConf, viewer) -> None:
        self.conf = conf
        self.next_entity_id = 0
        self.tracks: List[Track] = []
        self.viewer = viewer

    # ---- math helpers ----
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
        c = cluster.geometry.centroid
        R = cluster.geometry.rotation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = c
        return T

    def _shape_rel_dev(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.maximum(1e-6, 0.5 * (np.abs(a) + np.abs(b)))
        return float(np.mean(np.abs(a - b) / denom))

    def _gated_cost(self, T_pred, size_pred, T_det, size_det) -> float:
        d_pose = pose_distance_SE3(T_pred, T_det, self.conf.m_per_rad)
        d_shape = self._shape_rel_dev(size_pred, size_det)
        if d_pose > self.conf.max_dist or d_shape > self.conf.shape_gate:
            return np.inf
        return d_pose + self.conf.w_shape * d_shape

    def _estimate_twist_from_pair(
        self, T_prev: np.ndarray, T_now: np.ndarray, dt: float
    ) -> np.ndarray:
        # T_rel = T_prev^{-1} T_now, then (omega, v) = log_SE3(T_rel)
        # Per-second twist hat: xi^ = hat_se3(omega/dt, v/dt)
        T_rel = np.linalg.inv(T_prev) @ T_now
        omega, v = log_SE3(T_rel)
        dt_safe = max(float(dt), 1e-6)
        return hat_se3(omega / dt_safe, v / dt_safe)

    # ---- public helpers ----
    def get_predictions(self, dt: float) -> List[np.ndarray]:
        return [
            self._prediction(tr.state.pose, tr.state.twist, dt) for tr in self.tracks
        ]

    def solve_assignment(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if C.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        r, c = linear_sum_assignment(C)
        return r, c

    # ---- matching & lifecycle ----
    def _build_cost_matrix(self, detections: List[Cluster], dt: float) -> np.ndarray:
        print(len(self.tracks))
        preds = self.get_predictions(dt)
        C = np.full((len(self.tracks), len(detections)), np.inf, dtype=float)
        print(C.shape)
        for i, (tr, T_pred) in enumerate(zip(self.tracks, preds)):
            for j, cl in enumerate(detections):
                T_det = self._extract_pose(cl)
                C[i, j] = self._gated_cost(
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

            if tr.last_pose is None:
                tr.last_pose = tr.state.pose

            twist_hat = self._estimate_twist_from_pair(tr.state.pose, T_meas, dt)
            tr.state.twist = twist_hat

            tr.state.pose = T_meas
            tr.state.shape_size = cl.geometry.sizes
            tr.timestamp += dt
            tr.last_pose = T_meas
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
            twist0 = np.zeros((4, 4), dtype=float)
            st = self._make_state(T0, twist0, cl.geometry.sizes)
            self.tracks.append(
                Track(
                    entity_id=self.next_entity_id,
                    state=st,
                    timestamp=float(t0),
                    last_pose=T0.copy(),
                    missed=0,
                )
            )
            self.next_entity_id += 1

    def _prune_dead(self):
        self.tracks = [tr for tr in self.tracks if tr.missed <= self.conf.missed_tracks]

    # ---- main ----
    def apply(self, scenes: List[Scene]):
        if not scenes:
            return

        # init from first scene
        first = scenes[0]
        t_first = float(first.timestamp or 0.0)
        self.tracks = []
        for cl in first.scene_clusters or []:
            T0 = self._extract_pose(cl)
            twist0 = np.zeros((4, 4), dtype=float)
            st = self._make_state(T0, twist0, cl.geometry.sizes)
            self.tracks.append(
                Track(
                    entity_id=self.next_entity_id,
                    state=st,
                    timestamp=t_first,
                    last_pose=T0.copy(),
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
            dt = float(next_scene.timestamp - current_scene.timestamp)
            if not np.isfinite(dt) or dt <= 0:
                dt = self.conf.dt_default

            detections = next_scene.scene_clusters or []

            # build cost, solve, filter invalid (inf) pairs
            C = self._build_cost_matrix(detections, dt)
            print(C)
            M = 1e6
            inf_indices = np.where(C == np.inf)
            C[inf_indices] = M
            print(C)
            rows, cols = self.solve_assignment(C)
            pairs = [
                (i, j) for i, j in zip(rows.tolist(), cols.tolist()) if C[i, j] < M
            ]

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

            # self.viewer.add_matches(
            #     prev_scene_index=k - 1,
            #     next_scene_index=k,
            #     assignments=pairs,
            #     predicted_poses=preds,
            #     track_ids=[tr.entity_id for tr in self.tracks],
            #     next_clusters=next_scene.scene_clusters,
            #     predicted_sizes=pred_sizes,
            # )

            current_scene = next_scene
