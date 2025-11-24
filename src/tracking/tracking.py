from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from lidar_types import Scene, Cluster
from tracking.config import TrackingConf
from tracking.pose_calculations import (
    ctrv_jacobian_analytic,
    se2_state_from_T,
    ctrv_process_model,
    wrap_angle,
    T_from_se2_state,
    meas_from_pose,
    measurement_noise_R,
    extract_pose,
    shape_rel_dev,
)


@dataclass
class Observation:
    """
    Represents a cluster observation associated with a track at a given time.

    We store:
      - the detection index in the scene,
      - the member point indices,
      - the filtered SE(3) pose (T_w),
      - the filtered EKF state x_filt,
      - the cluster's size and timestamp.
    """

    scene_idx: int
    det_index: int
    member_indices: np.ndarray
    T_w: np.ndarray  # (4,4) SE(3) pose in world coords (filtered)
    x_filt: np.ndarray  # (5,) EKF state at this time
    sizes: np.ndarray  # (3,)
    timestamp: float


@dataclass
class TrackState:
    """
    EKF state for a single tracked object.

    State vector:
        x = [px, py, yaw, v, omega]
    Covariance:
        P = 5x5 covariance matrix

    pose_template:
        4x4 SE(3) transform used as a template for embedding SE(2) state
        into SE(3): its z/roll/pitch are used and only px,py,yaw are
        taken from x.
    """

    x: np.ndarray  # (5,)
    P: np.ndarray  # (5, 5)
    pose_template: np.ndarray  # (4, 4)
    shape_size: np.ndarray  # (3,)


@dataclass
class Track:
    """
    Represents the continued evolution of a single object across LiDAR frames.
    """

    entity_id: int
    state: TrackState
    timestamp: float
    missed: int = 0  # consecutive frames without an association
    hits: int = 0  # total successful associations
    is_confirmed: bool = False  # true once we trust the track
    observations: list[Observation] = field(default_factory=list)

    @property
    def pose(self) -> np.ndarray:
        """
        Derived SE(3) pose from EKF state + pose_template.
        """
        return T_from_se2_state(self.state.x, self.state.pose_template)


@dataclass
class TrackingResult:
    """
    Final output of the tracker:

    - det_to_track_per_scene: for each scene, list mapping detection index -> track ID
      (or -1 if unassigned)
    - point_to_entity_per_scene: for each scene, array of length #points mapping
      point index -> entity ID (or -1 if unassigned). Optional.
    - tracks: the list of all tracks (tentative + confirmed); consumers can filter
      by is_confirmed if needed.
    """

    det_to_track_per_scene: List[List[int]] = field(default_factory=list)
    point_to_entity_per_scene: Optional[List[np.ndarray]] = None
    tracks: List[Track] = field(default_factory=list)


class Tracker:
    def __init__(self, conf: TrackingConf) -> None:
        self.conf = conf
        self.next_entity_id: int = 0
        self.tracks: List[Track] = []

        # "infinite" cost for Hungarian
        self.M: float = 1e9

        # Linear measurement model: z = H x
        # z = [px, py, yaw]
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],  # px
                [0.0, 1.0, 0.0, 0.0, 0.0],  # py
                [0.0, 0.0, 1.0, 0.0, 0.0],  # yaw
            ],
            dtype=float,
        )
        self.I5 = np.eye(5, dtype=float)

        # lifecycle configuration (safe defaults via getattr)
        self.min_confirm_hits: int = getattr(self.conf, "min_confirm_hits", 2)
        self.tentative_prune: int = getattr(self.conf, "tentative_prune", 2)
        self.verbose: bool = getattr(self.conf, "verbose", False)

    def _make_state(self, T0: np.ndarray, shape_size: np.ndarray) -> TrackState:
        """
        Create an initial TrackState from an SE3 pose and object size.
        """
        # initial state [px, py, yaw, v, omega]
        x0 = se2_state_from_T(T0, v=0.0, omega=0.0)

        # Initial uncertainty in state:
        P0 = (
            np.diag(
                [
                    1.0,  # px
                    1.0,  # py
                    np.deg2rad(20.0),  # yaw
                    5.0,  # v
                    np.deg2rad(45.0),  # omega
                ]
            )
            ** 2
        )

        return TrackState(
            x=x0,
            P=P0,
            pose_template=T0.copy(),  # z/roll/pitch embedded here
            shape_size=shape_size.copy(),
        )

    def _pose_from_state(self, state: TrackState) -> np.ndarray:
        """
        Derived SE(3) pose for a given TrackState.
        """
        return T_from_se2_state(state.x, state.pose_template)

    def get_predictions(self, dt: float) -> List[np.ndarray]:
        """
        EKF prediction step for all tracks.

        Returns list of predicted full SE(3) poses.
        """
        preds: List[np.ndarray] = []

        # Additional uncertainty in prediction (process noise)
        Q = np.diag(
            [
                self.conf.q_pos**2,  # px
                self.conf.q_pos**2,  # py
                self.conf.q_yaw**2,  # yaw
                self.conf.q_v**2,  # v
                self.conf.q_omega**2,  # omega
            ]
        )

        for tr in self.tracks:
            x = tr.state.x
            P = tr.state.P

            # predict state using CTRV process model
            x_pred = ctrv_process_model(x, dt)
            x_pred[2] = wrap_angle(x_pred[2])

            F = ctrv_jacobian_analytic(x, dt)

            # propagate covariance
            P_pred = F @ P @ F.T + Q

            tr.state.x = x_pred
            tr.state.P = P_pred

            preds.append(self._pose_from_state(tr.state))

        return preds

    @staticmethod
    def solve_assignment(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the assignment of detections to tracks given cost matrix C.
        """
        r, c = linear_sum_assignment(C)
        return r, c

    def _build_cost_matrix(self, detections: List[Cluster]) -> np.ndarray:
        """
        Build cost matrix between predicted tracks and current detections.

        Cost is sqrt(Mahalanobis^2) + w_shape * relative_size_deviation,
        with gating in Mahalanobis space, relative shape, and Euclidean XY gating.
        """
        nT = len(self.tracks)
        nD = len(detections)
        C = np.full((nT, nD), self.M, dtype=float)

        if nT == 0 or nD == 0:
            return C

        # Measurement noise
        R = measurement_noise_R(self.conf.r_pos, self.conf.r_yaw)

        # Pre-compute detected poses and measurements
        detected_poses = [extract_pose(cl) for cl in detections]
        detected_meas = [meas_from_pose(T) for T in detected_poses]

        for i, tr in enumerate(self.tracks):
            x_pred = tr.state.x
            P_pred = tr.state.P

            # predicted measurement mean
            z_pred = self.H @ x_pred

            # innovation covariance
            S = self.H @ P_pred @ self.H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # if S is singular, skip this track (keep infinite costs)
                continue

            for j, z_det in enumerate(detected_meas):
                # innovation
                y = z_det - z_pred
                y[2] = wrap_angle(y[2])

                # Euclidean gating in XY (coarse)
                d_xy = np.linalg.norm(y[:2])
                if d_xy > self.conf.max_dist:
                    continue

                # Mahalanobis distance and gating
                d2 = float(y.T @ (S_inv @ y))
                if d2 > self.conf.maha_thresh:
                    continue

                # shape mismatch gate
                cl = detections[j]
                d_shape = shape_rel_dev(tr.state.shape_size, cl.geometry.sizes)
                if d_shape > self.conf.shape_gate:
                    continue

                # final cost
                C[i, j] = np.sqrt(d2) + self.conf.w_shape * d_shape

        return C

    def _handle_unmatched_tracks(self, unmatched_tracks: List[int]) -> None:
        """
        Increase missed count for unmatched tracks.
        """
        for i in unmatched_tracks:
            self.tracks[i].missed += 1

    def _update_matched(
        self,
        pairs: List[Tuple[int, int]],
        detections: List[Cluster],
        scene_idx: int,
        scene_timestamp: float,
    ) -> None:
        """
        EKF update for matched track-detection pairs and observation logging.
        """
        R = measurement_noise_R(self.conf.r_pos, self.conf.r_yaw)

        for i, j in pairs:
            tr = self.tracks[i]
            cl = detections[j]

            # measurement from cluster
            T_meas = extract_pose(cl)
            z = meas_from_pose(T_meas)

            # EKF update
            x_pred = tr.state.x
            P_pred = tr.state.P

            z_pred = self.H @ x_pred
            y = z - z_pred
            y[2] = wrap_angle(y[2])

            S = self.H @ P_pred @ self.H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # skip update if S is singular
                continue

            K = P_pred @ self.H.T @ S_inv

            x_new = x_pred + K @ y
            x_new[2] = wrap_angle(x_new[2])
            P_new = (self.I5 - K @ self.H) @ P_pred

            tr.state.x = x_new
            tr.state.P = P_new
            tr.state.shape_size = cl.geometry.sizes.copy()

            tr.timestamp = scene_timestamp
            tr.missed = 0
            tr.hits += 1

            # promote tentative → confirmed
            if not tr.is_confirmed and tr.hits >= self.min_confirm_hits:
                tr.is_confirmed = True

            # store observation with filtered pose/state
            T_filt = self._pose_from_state(tr.state)
            member_idx = np.array(getattr(cl, "member_indices", []), dtype=int)

            tr.observations.append(
                Observation(
                    scene_idx=scene_idx,
                    det_index=j,
                    member_indices=member_idx,
                    T_w=T_filt.copy(),
                    x_filt=x_new.copy(),
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
    ) -> None:
        """
        Spawn tentative tracks from unmatched detections with enough points.
        """
        for j in unmatched_dets:
            cl = detections[j]
            member_idx = np.array(getattr(cl, "member_indices", []), dtype=int)
            npts = len(member_idx)
            if npts < self.conf.min_points:
                continue

            T0 = extract_pose(cl)
            st = self._make_state(T0, cl.geometry.sizes)

            tr = Track(
                entity_id=self.next_entity_id,
                state=st,
                timestamp=t0,
                missed=0,
                hits=1,  # first hit (creation)
                is_confirmed=False,  # start as tentative
            )

            # initial observation uses the initial state
            tr.observations.append(
                Observation(
                    scene_idx=scene_idx,
                    det_index=j,
                    member_indices=member_idx,
                    T_w=T0.copy(),
                    x_filt=st.x.copy(),
                    sizes=cl.geometry.sizes.copy(),
                    timestamp=t0,
                )
            )

            self.tracks.append(tr)
            self.next_entity_id += 1

    def _prune_dead(self) -> None:
        """
        Remove tracks that have been missed for too long.

        - Confirmed tracks survive longer (self.conf.missed_tracks).
        - Tentative tracks are pruned quickly (self.tentative_prune).
        """
        new_tracks: List[Track] = []
        for tr in self.tracks:
            if tr.is_confirmed:
                if tr.missed < self.conf.missed_tracks:
                    new_tracks.append(tr)
            else:
                # tentative tracks pruned aggressively
                if tr.missed < self.tentative_prune and tr.hits > 0:
                    new_tracks.append(tr)
        self.tracks = new_tracks

    def _build_point_to_entity_mapping(
        self, scene: Scene, det_to_track: List[int]
    ) -> np.ndarray:
        """
        Build array mapping each point index -> entity ID for a given scene.

        Points associated with no track get ID -1.
        """
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

    def fit(self, scenes: List[Scene]) -> TrackingResult:
        """
        Apply tracking to a list of scenes.
        """
        result = TrackingResult(
            det_to_track_per_scene=[],
            point_to_entity_per_scene=[] if self.conf.build_point_map else None,
            tracks=[],
        )

        if not scenes:
            return result

        first = scenes[0]
        t_first = float(first.timestamp or 0.0)
        self.tracks = []
        self.next_entity_id = 0

        for det_idx, cl in enumerate(first.scene_clusters or []):
            T0 = extract_pose(cl)
            st = self._make_state(T0, cl.geometry.sizes)

            tr = Track(
                entity_id=self.next_entity_id,
                state=st,
                timestamp=t_first,
                missed=0,
                hits=1,
                is_confirmed=True,  # first-frame objects are trusted
            )

            member_idx = np.array(getattr(cl, "member_indices", []), dtype=int)
            tr.observations.append(
                Observation(
                    scene_idx=0,
                    det_index=det_idx,
                    member_indices=member_idx,
                    T_w=T0.copy(),
                    x_filt=st.x.copy(),
                    sizes=cl.geometry.sizes.copy(),
                    timestamp=t_first,
                )
            )

            self.tracks.append(tr)
            self.next_entity_id += 1

        # det -> track map for scene 0 (direct births in order)
        det_to_track_0 = [-1] * len(first.scene_clusters or [])
        for i, cl in enumerate(first.scene_clusters or []):
            if i < len(self.tracks):
                det_to_track_0[i] = self.tracks[i].entity_id

        result.det_to_track_per_scene.append(det_to_track_0)
        if self.conf.build_point_map and result.point_to_entity_per_scene is not None:
            result.point_to_entity_per_scene.append(
                self._build_point_to_entity_mapping(first, det_to_track_0)
            )

        if len(scenes) == 1:
            result.tracks = list(self.tracks)
            return result

        current_scene = scenes[0]

        for k, next_scene in enumerate(scenes[1:], start=1):
            if next_scene.scene_clusters is None:
                # still advance result slots so lengths match 'scenes'
                result.det_to_track_per_scene.append([])
                if (
                    self.conf.build_point_map
                    and result.point_to_entity_per_scene is not None
                ):
                    result.point_to_entity_per_scene.append(np.array([], dtype=int))
                current_scene = next_scene
                continue

            assert current_scene.timestamp is not None
            assert next_scene.timestamp is not None
            dt = float(next_scene.timestamp - current_scene.timestamp)
            if not np.isfinite(dt) or dt <= 0:
                dt = self.conf.dt_default

            # 1) predict all tracks to next timestamp
            self.get_predictions(dt)

            # 2) build cost matrix & solve assignment
            detections = next_scene.scene_clusters or []
            C = self._build_cost_matrix(detections)
            rows, cols = self.solve_assignment(C)

            # collect valid pairs (cost < M)
            pairs = [
                (i, j) for i, j in zip(rows.tolist(), cols.tolist()) if C[i, j] < self.M
            ]

            if self.verbose:
                print(
                    f"[Frame {k}] #tracks={len(self.tracks)} "
                    f"#dets={len(detections)} #pairs={len(pairs)}"
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

            # 3) lifecycle updates
            self._update_matched(
                pairs,
                detections,
                scene_idx=k,
                scene_timestamp=float(next_scene.timestamp),
            )
            self._handle_unmatched_tracks(unmatched_tracks)
            self._spawn_tracks(
                detections, unmatched_dets, t0=float(next_scene.timestamp), scene_idx=k
            )
            self._prune_dead()

            # 4) Fix mapping for detections that spawned new tracks this frame
            born_ids = {
                tr.entity_id
                for tr in self.tracks
                if abs(tr.timestamp - float(next_scene.timestamp)) < 1e-9
            }
            if any(eid == -1 for eid in det_to_track):
                for j, eid in enumerate(det_to_track):
                    if eid != -1:
                        continue
                    for tr in self.tracks:
                        if tr.entity_id not in born_ids:
                            continue
                        if not tr.observations:
                            continue
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

            current_scene = next_scene

        result.tracks = list(self.tracks)
        return result
