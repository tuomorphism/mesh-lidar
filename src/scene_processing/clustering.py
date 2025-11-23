import numpy as np
from lidar_types import ClusterGeometry, Scene, Cluster
from scene_processing.config import Config
from scene_processing.scanning import DbscanConfig, dbscan_3d
from sklearn.cluster import KMeans


def compute_yaw_obb(points: np.ndarray, eps: float = 1e-6):

    def _yaw_from_cov_xy(XY: np.ndarray) -> float:
        # XY: (N,2) centered points
        C = XY.T @ XY / max(XY.shape[0] - 1, 1)
        # principal dir is eigenvector of largest eigval
        _, vecs = np.linalg.eigh(C)  # eigh: symmetric -> sorted ascending
        v = vecs[:, -1]  # principal axis in XY
        # yaw from principal axis
        return float(np.arctan2(v[1], v[0]))

    def _R_from_yaw(yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.eye(3)
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c
        return R

    P = np.asarray(points, float)
    assert P.ndim == 2 and P.shape[1] == 3
    N = P.shape[0]
    if N == 0:
        raise ValueError("No points for OBB")

    center = P.mean(axis=0)
    X = P - center

    # yaw from XY covariance (ignoring z for orientation)
    yaw = _yaw_from_cov_xy(X[:, :2])
    Rz = _R_from_yaw(yaw)

    Y = X @ Rz
    lo = Y.min(axis=0)
    hi = Y.max(axis=0)
    extents = np.maximum(hi - lo, eps)

    local_center = 0.5 * (lo + hi)
    lo -= local_center
    hi -= local_center
    center_world = center + local_center @ Rz.T

    corners_local = _corners_from_lo_hi(lo, hi)
    corners_world = _world_from_local(corners_local, center_world, Rz)

    return corners_world, center_world, Rz, extents, lo, hi, yaw


def _corners_from_lo_hi(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Return 8 corners in a consistent order from per-axis lo/hi in local coords."""
    xs = [lo[0], hi[0]]
    ys = [lo[1], hi[1]]
    zs = [lo[2], hi[2]]
    corners = [
        [xs[0], ys[0], zs[0]],
        [xs[1], ys[0], zs[0]],
        [xs[0], ys[1], zs[0]],
        [xs[0], ys[0], zs[1]],
        [xs[1], ys[1], zs[0]],
        [xs[1], ys[0], zs[1]],
        [xs[0], ys[1], zs[1]],
        [xs[1], ys[1], zs[1]],
    ]
    return np.asarray(corners, dtype=np.float64)


def _world_from_local(
    corners_local: np.ndarray, center: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Map local corners → world via x_world = center + R @ x_local."""
    return corners_local @ R.T + center


def _compute_cluster_geometry(cluster_data: np.ndarray) -> ClusterGeometry:
    assert (
        cluster_data.shape[1] == 4
    ), f"cluster data has shape {cluster_data.shape} instead of (N, 4)!"
    # Separate the 3d data from other variables
    cluster_points = cluster_data[:, :3]

    corners, center, Rz, sizes, low_points, high_points, yaw = compute_yaw_obb(
        cluster_points
    )

    cov = (
        (cluster_points - center).T
        @ (cluster_points - center)
        / max(cluster_points.shape[0] - 1, 1)
    )
    mean_intensity = cluster_data[:, 3].mean()

    return ClusterGeometry(
        centroid=center,
        bbox=corners,
        mean_intensity=mean_intensity,
        cov=cov,
        rotation=Rz,
        sizes=sizes,
    )


def merge_close_clusters(scene: Scene) -> Scene:
    clusters = scene.scene_clusters or []
    if not clusters:
        return scene

    pts = scene.points
    N = len(clusters)

    v_meds = np.zeros((N,))

    assert scene.velocity_field is not None
    for i, cl in enumerate(clusters):
        v = scene.velocity_field[cl.member_indices]
        speeds = np.linalg.norm(v[:, :2], axis=1)
        v_meds[i] = np.median(speeds)

    centers = np.array([c.geometry.centroid for c in clusters])

    # adjacency matrix for merging
    adj = np.zeros((N, N), dtype=bool)

    for i in range(N):
        for j in range(i + 1, N):
            d_xy = np.linalg.norm(centers[i][:2] - centers[j][:2])
            if d_xy > Config.merge_gap_threshold:
                continue

            dv = abs(v_meds[i] - v_meds[j])
            if dv > Config.merge_speed_thr:
                continue

            # if all criteria pass → connect them
            adj[i, j] = adj[j, i] = True

    # --- connected components merging ---
    visited = np.zeros(N, dtype=bool)
    merged = []

    for i in range(N):
        if visited[i]:
            continue

        stack = [i]
        comp = []
        visited[i] = True

        while stack:
            k = stack.pop()
            comp.append(k)
            for j in np.where(adj[k])[0]:
                if not visited[j]:
                    visited[j] = True
                    stack.append(j)

        # merge comp
        member_idx = np.concatenate([clusters[k].member_indices for k in comp])
        pts_comp = pts[member_idx]
        geom = _compute_cluster_geometry(pts_comp)
        merged.append(Cluster(member_indices=member_idx.tolist(), geometry=geom))

    # replace scene clusters
    scene.scene_clusters = merged
    return scene


def split_cluster_by_velocity(cluster: Cluster, scene: Scene):
    """
    If the cluster contains both static and moving points,
    split it into subclusters using velocity magnitude.
    """
    assert scene.velocity_field is not None, "scene should have velocity field!"
    idx = np.array(cluster.member_indices)
    v = scene.velocity_field[idx][:, :2]
    speeds = np.linalg.norm(v, axis=1)

    static_mask = speeds < Config.static_speed_thr
    moving_mask = speeds > Config.moving_speed_thr

    # Case 1: pure cluster, don't split
    if static_mask.all() or moving_mask.all():
        return [cluster]

    # Case 2: mixed: split into at least 2 groups

    km = KMeans(n_clusters=2, n_init=5)
    labels = km.fit_predict(speeds.reshape(-1, 1))

    # Build new clusters
    new_clusters = []
    for lbl in [0, 1]:
        sub_idx = idx[labels == lbl]
        if sub_idx.size == 0:
            continue
        geom = _compute_cluster_geometry(scene.points[sub_idx])
        new_clusters.append(Cluster(member_indices=sub_idx.tolist(), geometry=geom))
    return new_clusters


def compute_clusters_geom(
    scene: Scene,
) -> Scene:
    pts = scene.points
    if pts.size == 0:
        return scene

    xyz = pts[:, :3]
    scaled_xyz = xyz * Config.clustering_scale

    labels = dbscan_3d(
        scaled_xyz,
        DbscanConfig(
            eps=Config.eps_factor * Config.voxel_size,
            min_samples=Config.min_samples,
            leaf_size=40,
            n_jobs=-1,
        ),
    )

    clusters: list[Cluster] = []
    for c_id in np.unique(labels):
        if c_id == -1:
            continue  # DBSCAN "noise"
        idx = np.where(labels == c_id)[0]
        if idx.size < Config.min_samples:
            continue  # tiny garbage, OK to drop

        geom = _compute_cluster_geometry(pts[idx])
        clusters.append(
            Cluster(
                member_indices=idx.tolist(),
                geometry=geom,
            )
        )

    split_clusters = []
    for cl in clusters:
        split_clusters.extend(split_cluster_by_velocity(cl, scene))

    clustered_scene = Scene(
        points=scene.points,
        ground_plane=scene.ground_plane,
        scene_clusters=clusters,
        timestamp=scene.timestamp,
        velocity_field=scene.velocity_field,
    )
    merged_clusters = merge_close_clusters(clustered_scene)
    return merged_clusters
