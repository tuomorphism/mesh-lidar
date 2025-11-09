import json

import numpy as np
import open3d as o3d

from lidar_types import Scene, Cluster, Sweep
from clustering import compute_clusters


def preprocess(points: np.ndarray, voxel_size=0.10) -> Scene:
    """
    preprocessing of sweep points before clustering
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Estimating ground
    ground_plane, ground_points = pcd.segment_plane(
        distance_threshold=5 * voxel_size, ransac_n=20, num_iterations=100
    )
    mask = np.ones(points.shape[0], dtype=bool)
    mask[ground_points] = False

    non_ground_points = points[mask, :]

    return Scene(points=non_ground_points, ground_plane=ground_plane)


def postprocess(scene: Scene) -> Scene:
    """
    postprocessing of scene after clustering
    """

    def _filter_cluster(cluster: Cluster) -> bool:
        # Simple variance of X, Y and Z
        if cluster.geometry.cov.diagonal().sum() > 100:
            return False
        return True

    filtered_clusters = []
    for cluster in scene.scene_clusters or []:
        if _filter_cluster(cluster) is True:
            filtered_clusters.append(cluster)

    scene = Scene(
        points=scene.points,
        ground_plane=scene.ground_plane,
        scene_clusters=filtered_clusters,
    )
    return scene


def _process_sweep(
    index: int, sweep: np.ndarray, metadata: dict | None = None
) -> Scene:
    preprocessed = preprocess(sweep)
    clustered = compute_clusters(preprocessed)
    postprocessed = postprocess(clustered)
    postprocessed.timestamp = (
        metadata.get("timestamp", index) * 0.1
        if metadata is not None
        else index * 0.1  # Assumes 10Hz sweeps.
    )
    return postprocessed


def process_sweeps(sweeps: list[Sweep]) -> list[Scene]:
    """
    Processes a list of sweep data into Scene objects.
    """
    return list(
        map(
            lambda x: _process_sweep(x[0], x[1].pts, x[1].metadata),
            enumerate(sweeps),
        )
    )
