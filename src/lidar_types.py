from dataclasses import dataclass, field
from typing import Optional
import numpy as np


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
