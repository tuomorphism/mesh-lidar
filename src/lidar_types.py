from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class Sweep:
    """
    Represents a LiDAR sweep.

    pts: (N, 4) or (N, D) ndarray
        Columns: [x, y, z, ...] where extra columns can be intensity/ring/etc.
    metadata: dict
        Arbitrary metadata, expected to contain at least:
        - 'timestamp_ms'
        - 'sensor'
        - extrinsics info (see below)
    """

    pts: np.ndarray
    metadata: Dict[str, Any]

    @property
    def frame(self) -> str:
        """
        Name of the coordinate frame this sweep's points are expressed in.

        Defaults to 'sensor' if not present.
        """
        return self.metadata.get("frame", "sensor")


@dataclass
class ClusterGeometry:
    """
    Dataclass for cluster geometry
    """

    centroid: np.ndarray
    bbox: np.ndarray
    mean_intensity: float
    cov: np.ndarray
    rotation: np.ndarray
    sizes: np.ndarray


@dataclass
class Cluster:
    """
    Dataclass describing a 3d cluster
    """

    member_indices: list[int]
    geometry: ClusterGeometry
    label: int = field(default=0)
    velocity_xy: np.ndarray | None = None

    def points(self, raw_data: np.ndarray) -> np.ndarray:
        """
        points belonging to the cluster
        """
        return raw_data[self.member_indices, :]


@dataclass
class Scene:
    """
    Main 3d scene dataclass
    """

    points: np.ndarray
    ground_plane: np.ndarray
    scene_clusters: Optional[list[Cluster]] = None
    timestamp: Optional[float] = None
    velocity_field: Optional[np.ndarray] = None

    @property
    def cluster_membership(self) -> dict[int, Cluster]:
        """
        membership of each point index with the cluster
        """
        if self.scene_clusters is None:
            return {}
        total_membership_mapping = {}
        for cluster in self.scene_clusters:
            cluster_mapping = {idx: cluster for idx in cluster.member_indices}
            total_membership_mapping = total_membership_mapping | cluster_mapping
        return total_membership_mapping

    @property
    def cluster_labels(self) -> np.ndarray:
        """
        simple list of cluster labels for all points
        """

        def _map_cluster(cluster: Cluster | None):
            if cluster is None:
                return -1
            return cluster.label

        mapping = self.cluster_membership
        return np.asarray(
            [_map_cluster(mapping.get(idx)) for idx in range(self.points.shape[0])]
        )


@dataclass
class TrackSnapshot:
    """
    Per-frame snapshot of a tracked object.

    P is the EKF covariance at this time, used for uncertainty visualization.
    """

    timestamp: float
    scene_idx: int
    entity_id: int

    T_w: np.ndarray  # (4,4) SE(3) pose
    x_filt: np.ndarray  # (5,) [px, py, yaw, v, omega]
    P: np.ndarray  # (5,5) covariance at this frame

    sizes: np.ndarray  # (3,)
    member_indices: np.ndarray  # indices into the scene point cloud


@dataclass
class TrackHistory:
    """
    Full time-series for a single tracked entity.
    """

    entity_id: int
    is_confirmed: bool
    snapshots: List[TrackSnapshot]

    mean_speed: float
    is_static: bool
