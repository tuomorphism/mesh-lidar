import numpy as np
import open3d as o3d

from lidar_types import Scene, Cluster, Sweep
from scene_processing.clustering import compute_clusters_geom
from scene_processing.velocity_filter import nn_flow
from scene_processing.config import Config


def voxel_downsample_with_intensity(points):
    xyz = points[:, :3]
    feat = points[:, 3:]

    coords = np.floor(xyz / Config.voxel_size).astype(np.int32)
    keys = (
        coords[:, 0].astype(np.int64) * 73856093
        + coords[:, 1].astype(np.int64) * 19349663
        + coords[:, 2].astype(np.int64) * 83492791
    )

    uniq_keys, inv = np.unique(keys, return_inverse=True)
    M = len(uniq_keys)

    ds_xyz = np.zeros((M, 3), dtype=np.float32)
    ds_feat = np.zeros((M, feat.shape[1]), dtype=np.float32)

    np.add.at(ds_xyz, inv, xyz)  # accumulate xyz sums
    np.add.at(ds_feat, inv, feat)  # accumulate feature sums

    counts = np.bincount(inv).astype(np.float32)
    ds_xyz /= counts[:, None]
    ds_feat /= counts[:, None]

    return np.hstack([ds_xyz, ds_feat])


def estimate_normals_o3d(
    pcd: o3d.geometry.PointCloud,
    radius: float = 0.5,
    max_nn: int = 30,
):
    """Estimate normals using Open3D KD-tree."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.normalize_normals()
    return pcd


def preprocess(
    i: int,
    points: np.ndarray,
    metadata: dict = {},
) -> Scene:
    """
    Downsample [x,y,z,i,...], remove ground with RANSAC, return Scene with DS non-ground.
    """
    assert points.ndim == 2 and points.shape[1] >= 3
    print(f"Preprocessing index {i}.")

    ds_points = voxel_downsample_with_intensity(points)
    if ds_points.shape[0] == 0:
        D = points.shape[1]
        return Scene(
            points=np.empty((0, D)),
            ground_plane=np.empty((0, D)),
            timestamp=metadata.get("timestamp"),
        )

    pcd_ds = o3d.geometry.PointCloud()
    pcd_ds.points = o3d.utility.Vector3dVector(ds_points[:, :3])

    _, inliers = pcd_ds.segment_plane(
        distance_threshold=0.4,
        ransac_n=5,
        num_iterations=300,
    )

    inliers = np.asarray(inliers, dtype=int)
    mask = np.zeros(ds_points.shape[0], dtype=bool)
    mask[inliers] = True

    ds_ground = ds_points[mask]
    ds_non_ground = ds_points[~mask]

    return Scene(
        points=ds_non_ground,  # [x,y,z,intensity,...] non-ground, already DS
        ground_plane=ds_ground,
        timestamp=metadata.get("timestamp"),
    )


def postprocess(scene: Scene) -> Scene:
    """
    - Drop only very small clusters (noise).
    - Check the Z extent of cluster
    """
    if scene.scene_clusters is None:
        return scene

    filtered: list[Cluster] = []
    for c in scene.scene_clusters:
        n = len(c.member_indices)
        if n < Config.min_samples:  # tiny splats, OK to drop
            continue

        if c.geometry.sizes[2] < Config.min_height_cutoff:
            continue

        filtered.append(c)

    return Scene(
        points=scene.points,
        ground_plane=scene.ground_plane,
        scene_clusters=filtered,
        timestamp=scene.timestamp,
        velocity_field=scene.velocity_field,
    )


def _cluster_scene_pair(i: int, A: Scene, B: Scene) -> Scene:
    print(f"Processing pair {i}, {i + 1}")
    dt = (
        B.timestamp - A.timestamp
        if A.timestamp is not None and B.timestamp is not None
        else Config.delta_t_fallback
    )
    vel, _ = nn_flow(A.points[:, :3], B.points[:, :3], dt=dt)

    A.velocity_field = vel
    clustered = compute_clusters_geom(A)
    out = postprocess(clustered)
    return out


def process_sweeps(sweeps: list[Sweep]) -> list[Scene]:
    pre = [preprocess(i, s.pts, metadata=s.metadata) for i, s in enumerate(sweeps)]
    return [_cluster_scene_pair(i, pre[i], pre[i + 1]) for i in range(len(pre) - 1)]
