import numpy as np
import open3d as o3d

from lidar_types import Scene, Cluster, Sweep
from scene_processing.clustering import compute_clusters_flow
from scene_processing.velocity_filter import nn_flow


def preprocess(points: np.ndarray, voxel_size=0.10, metadata: dict = {}) -> Scene:
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

    return Scene(
        points=non_ground_points,
        ground_plane=ground_plane,
        timestamp=metadata.get("timestamp"),
    )


def postprocess(scene: Scene) -> Scene:
    """
    postprocessing of scene after clustering
    """

    def _filter_cluster(cluster: Cluster) -> bool:
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
        timestamp=scene.timestamp,
        velocity_field=scene.velocity_field,
    )
    return scene


def _cluster_scene_pair(
    _: int,
    current_preprocessed: Scene,
    next_preprocessed: Scene,
) -> Scene:

    assert hasattr(current_preprocessed, "timestamp") and hasattr(
        next_preprocessed, "timestamp"
    ), "scenes should have timestamps!"
    assert (
        current_preprocessed.timestamp is not None
        and next_preprocessed.timestamp is not None
    )

    dt = next_preprocessed.timestamp - current_preprocessed.timestamp

    if next_preprocessed is not None:
        vel_flow, valid_mask = nn_flow(
            current_preprocessed.points[:, :3],
            next_preprocessed.points[:, :3],
            dt=dt,
            max_dist=3,
        )
    else:
        vel_flow = np.zeros_like(current_preprocessed.points)
        valid_mask = np.ones_like(vel_flow, dtype=bool)

    clustered = compute_clusters_flow(current_preprocessed, vel_flow)
    postprocessed = postprocess(clustered)
    postprocessed.velocity_field = vel_flow
    return postprocessed


def process_sweeps(sweeps: list[Sweep]) -> list[Scene]:
    """
    Processes a list of sweep data into Scene objects.
    """

    preprocessed = list(map(lambda x: preprocess(x.pts, metadata=x.metadata), sweeps))
    clustered = list(
        map(
            lambda i: _cluster_scene_pair(i, preprocessed[i], preprocessed[i + 1]),
            range(len(preprocessed) - 1),
        )
    )
    postprocessed = list(map(postprocess, clustered))
    return postprocessed
