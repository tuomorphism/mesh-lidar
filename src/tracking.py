from dataclasses import dataclass, field
from os import pread
from typing import List, Tuple, Optional
from flask.cli import F
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
    lin_speed_from_twist,
    predict_se2_CTRV,
)
from tracking_plot import plot_scene_pair


@dataclass
class Observation:
    scene_idx: int
    det_index: int
    member_indices: np.ndarray
    T_w: np.ndarray
    sizes: np.ndarray
    timestamp: float


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
    observations: list[Observation] = field(default_factory=list)


@dataclass
class TrackingResult:
    det_to_track_per_scene: List[List[int]] = field(default_factory=list)
    point_to_entity_per_scene: Optional[List[np.ndarray]] = None
    tracks: List[Track] = field(default_factory=list)


@dataclass
class TrackingConf:
    missed_tracks: int = 8  # kill after this many consecutive misses
    min_points: int = 5  # min points to birth a new track
    max_dist: float = 3.0  # gating in meters (w/ m_per_rad coupling)
    m_per_rad: float = 0.6  # convert rad -> “meters” in pose metric
    w_shape: float = 0.1  # weight on relative size deviation
    shape_gate: float = 1.2  # max allowed avg relative size mismatch
    dt_default: float = 0.1  # fallback Δt
    build_point_map: bool = True


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
        """
        estimating the twist (element of lie algebra se(3)) from pair of world poses.
        """

        T_prev_sanitized = sanitize_T(T_prev)
        T_now_sanitized = sanitize_T(T_now)

        # Taking the inverse of previous position (sanitized) and then applying the current pose, we obtain the relative pose T_rel
        T_rel = rigid_inverse(T_prev_sanitized) @ T_now_sanitized

        # Now we take the Lie group logarithm of the relative pose, obtaining the twist in two components (omega and v)
        # omega is the rotation and v is the translation components
        omega, v = log_SE3(T_rel)
        dt_safe = max(float(dt), 1e-6)

        self.check_log_exp(T_rel)

        return hat_se3(omega / dt_safe, v / dt_safe)

    def get_predictions(self, dt: float) -> List[np.ndarray]:

        preds = []
        for tr in self.tracks:
            vx = tr.state.twist[0, 3]
            vy = tr.state.twist[1, 3]
            v = float(np.hypot(vx, vy))
            psi = float(np.arctan2(tr.state.pose[1, 0], tr.state.pose[0, 0]))
            psidot = float(tr.state.twist[1, 0])

            prediction = predict_se2_CTRV(tr.state.pose, v, psi, psidot, dt)
            preds.append(prediction)
        return preds

    def solve_assignment(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r, c = linear_sum_assignment(C)
        return r, c

    def _build_cost_matrix(self, detections: List[Cluster], dt: float) -> np.ndarray:
        preds = self.get_predictions(dt)
        C = np.full((len(self.tracks), len(detections)), self.M, dtype=float)

        detected_poses = [self._extract_pose(p) for p in detections]

        gamma = 1.2
        for i, (tr, T_pred) in enumerate(zip(self.tracks, preds)):
            # Two-stage pruning, first course distance screening
            v = lin_speed_from_twist(tr.state.twist)
            gate = max(self.conf.max_dist, 0.6 + 1.8 * v * dt)

            coarse_filter = []
            for j, T_det in enumerate(detected_poses):
                d = np.linalg.norm(T_pred[:3, 3] - T_det[:3, 3])
                if d <= gamma * gate:
                    coarse_filter.append(j)

            for j in coarse_filter:
                cl = detections[j]
                T_det = self._extract_pose(cl)
                C[i, j] = self._cost(
                    T_pred, tr.state.shape_size, T_det, cl.geometry.sizes
                )
        return C

    def _handle_unmatched(self, unmatched_tracks: List[int]):
        for i in unmatched_tracks:
            self.tracks[i].missed += 1

    def _update_matched(
        self,
        pairs: List[Tuple[int, int]],
        detections: List[Cluster],
        dt: float,
        scene_idx: int,
        scene_timestamp: float,
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

            member_idx = np.array(getattr(cl, "member_indices", []), dtype=int)
            tr.observations.append(
                Observation(
                    scene_idx=scene_idx,
                    det_index=j,
                    member_indices=member_idx,
                    T_w=T_meas.copy(),
                    sizes=cl.geometry.sizes.copy(),
                    timestamp=scene_timestamp,
                )
            )

    def _spawn_tracks(
        self,
        detections: List[Cluster],
        unmatched_dets: List[int],
        t0: float,
        scene_idx: int,
    ):
        for j in unmatched_dets:
            cl = detections[j]
            npts = len(getattr(cl, "member_indices", []))
            if npts < self.conf.min_points:
                continue
            T0 = self._extract_pose(cl)
            twist0 = np.zeros((4, 4))
            st = self._make_state(T0, twist0, cl.geometry.sizes)
            tr = Track(
                entity_id=self.next_entity_id,
                state=st,
                timestamp=t0,
                missed=0,
            )
            member_idx = np.array(getattr(cl, "member_indices", []), dtype=int)
            tr.observations.append(
                Observation(
                    scene_idx=scene_idx,
                    det_index=j,
                    member_indices=member_idx,
                    T_w=T0.copy(),
                    sizes=cl.geometry.sizes.copy(),
                    timestamp=t0,
                )
            )
            self.tracks.append(tr)
            self.next_entity_id += 1

    def _prune_dead(self):
        self.tracks = [tr for tr in self.tracks if tr.missed < self.conf.missed_tracks]

    def _build_point_to_entity_mapping(
        self, scene: Scene, det_to_track: List[int]
    ) -> np.ndarray:
        N = len(getattr(scene, "points", []))
        mapping = np.full(N, -1, dtype=int)
        if N == 0:
            return mapping
        clusters = scene.scene_clusters or []
        for det_idx, entity_id in enumerate(det_to_track):
            if entity_id < 0:
                continue
            if det_idx >= len(clusters):
                continue
            cl = clusters[det_idx]
            member_idx = getattr(cl, "member_indices", None)
            if member_idx is None:
                continue
            mapping[np.asarray(member_idx, dtype=int)] = int(entity_id)
        return mapping

    def apply(self, scenes: List[Scene]) -> TrackingResult:
        result = TrackingResult(
            det_to_track_per_scene=[],
            point_to_entity_per_scene=[] if self.conf.build_point_map else None,
            tracks=[],
        )

        if not scenes:
            return result

        # --- initialize from first scene ---
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

        # det -> track map for scene 0 (direct births in order)
        det_to_track = [-1] * len(first.scene_clusters or [])
        for i, cl in enumerate(first.scene_clusters or []):
            if i < len(self.tracks):
                det_to_track[i] = self.tracks[i].entity_id
                self.tracks[i].observations.append(
                    Observation(
                        scene_idx=0,
                        det_index=i,
                        member_indices=np.array(
                            getattr(cl, "member_indices", []), dtype=int
                        ),
                        T_w=self._extract_pose(cl),
                        sizes=cl.geometry.sizes,
                        timestamp=float(first.timestamp or 0.0),
                    )
                )

        result.det_to_track_per_scene.append(det_to_track)
        if self.conf.build_point_map and result.point_to_entity_per_scene is not None:
            result.point_to_entity_per_scene.append(
                self._build_point_to_entity_mapping(first, det_to_track)
            )

        if len(scenes) == 1:
            result.tracks = list(self.tracks)
            return result

        current_scene = scenes[0]

        # --- iterate remaining scenes ---
        for k, next_scene in enumerate(scenes[1:], start=1):
            if next_scene.scene_clusters is None:
                # still advance result slots so lengths match 'scenes'
                result.det_to_track_per_scene.append([])
                if (
                    self.conf.build_point_map
                    and result.point_to_entity_per_scene is not None
                ):
                    result.point_to_entity_per_scene.append(np.array([], dtype=int))
                continue

            assert current_scene.timestamp is not None
            assert next_scene.timestamp is not None
            dt = next_scene.timestamp - current_scene.timestamp
            if not np.isfinite(dt) or dt <= 0:
                dt = self.conf.dt_default

            detections = next_scene.scene_clusters or []
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

            # per-scene detection->entity mapping
            det_to_track = [-1] * len(detections)
            for i, j in pairs:
                det_to_track[j] = self.tracks[i].entity_id

            # lifecycle updates (now also recording observations)
            self._update_matched(
                pairs,
                detections,
                dt,
                scene_idx=k,
                scene_timestamp=float(next_scene.timestamp),
            )
            self._handle_unmatched(unmatched_tracks)
            self._spawn_tracks(
                detections, unmatched_dets, t0=float(next_scene.timestamp), scene_idx=k
            )
            self._prune_dead()

            # after spawns, some unmatched_dets may have new entity_ids; fill them in:
            # we fill by scanning detections and finding tracks that were just born at this scene
            # (lightweight heuristic: latest entity_ids with timestamp == next_scene.timestamp)
            born_ids = {
                tr.entity_id
                for tr in self.tracks
                if abs(tr.timestamp - float(next_scene.timestamp)) < 1e-9
            }
            if any(eid == -1 for eid in det_to_track):
                # try to bind remaining dets by matching observation (scene_idx=k, det_index=j)
                for j, eid in enumerate(det_to_track):
                    if eid != -1:
                        continue
                    cl = detections[j]
                    for tr in self.tracks:
                        if tr.entity_id in born_ids and tr.observations:
                            ob = tr.observations[-1]
                            if ob.scene_idx == k and ob.det_index == j:
                                det_to_track[j] = tr.entity_id
                                break

            result.det_to_track_per_scene.append(det_to_track)
            if (
                self.conf.build_point_map
                and result.point_to_entity_per_scene is not None
            ):
                result.point_to_entity_per_scene.append(
                    self._build_point_to_entity_mapping(next_scene, det_to_track)
                )

            # keep your existing viz
            plot_scene_pair(current_scene, next_scene, pairs)
            current_scene = next_scene

        result.tracks = list(self.tracks)
        return result
