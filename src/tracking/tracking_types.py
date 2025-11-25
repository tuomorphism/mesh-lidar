from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from tracking.pose_calculations import T_from_se2_state
from lidar_types import TrackHistory


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
    P_filt: np.ndarray  # (5, 5) covariance of the observation (uncertainty)
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
    last_det_index: Optional[int] = None

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
    - histories: visualization / reconstruction friendly per-track time series.
    """

    det_to_track_per_scene: List[List[int]] = field(default_factory=list)
    point_to_entity_per_scene: Optional[List[np.ndarray]] = None
    tracks: List[Track] = field(default_factory=list)
    histories: List[TrackHistory] = field(default_factory=list)
