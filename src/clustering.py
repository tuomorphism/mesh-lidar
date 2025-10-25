import numpy as np
from sklearn.cluster import DBSCAN
from lidar_types import ClusterGeometry, Scene, Cluster

def _cluster_scene_dbscan(points: np.ndarray, voxel_eps: float = 1.0) -> np.ndarray:
    print(voxel_eps)
    clustering = DBSCAN(eps = voxel_eps)
    clustering = clustering.fit(points)
    return clustering.labels_

def _compute_cluster_geometry(cluster_data: np.ndarray) -> ClusterGeometry:
    assert cluster_data.shape[1] == 5
    # Separate the 3d data from other variables
    cluster_points = cluster_data[:, :3]

    centroid = cluster_points.mean(axis=0)
    X = cluster_points - centroid
    cov = (X.T @ X) / (cluster_points.shape[0]-1)

    # Compute the local coordinate transform
    _, V = np.linalg.eigh(cov)
    Y = X @ V

    # Min and max in local coordinates
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    corners_local = np.array([
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], hi[1], lo[2]],
        [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]],
        [hi[0], hi[1], hi[2]]
    ])
    bbox = corners_local @ V.T + centroid
    mean_intensity = cluster_data[:, 3].mean()
    return ClusterGeometry(centroid, bbox, mean_intensity, cov)



def compute_clusters(scene: Scene) -> Scene:
    """
    Obtains clusters from a single scene. Returns a Scene object with set clusters
    """
    raw_clusters = _cluster_scene_dbscan(scene.points[:, :3], voxel_eps=0.8)
    cluster_labels = np.unique(raw_clusters)
    scene_clusters: list[Cluster] = []
    for c in cluster_labels:
        correct_mask = np.nonzero(raw_clusters == c)[0]
        if correct_mask.shape[0] == 0:
            continue
        cluster_points = scene.points[correct_mask, :]
        cluster_geo = _compute_cluster_geometry(cluster_points)
        cluster = Cluster(
            member_indices=correct_mask.tolist(),
            geometry=cluster_geo,
            label=c
        )

        scene_clusters.append(cluster)

    scene.scene_clusters = scene_clusters

    return scene