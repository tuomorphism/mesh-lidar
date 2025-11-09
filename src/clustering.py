import numpy as np
from sklearn.cluster import DBSCAN
from lidar_types import ClusterGeometry, Scene, Cluster


def _cluster_scene_dbscan(points: np.ndarray, voxel_eps: float = 1.0) -> np.ndarray:
    clustering = DBSCAN(eps=voxel_eps)
    clustering = clustering.fit(points)
    return clustering.labels_


def compute_pca_obb(points: np.ndarray, eps: float = 1e-9):
    """
    Compute an oriented bounding box for a 3D point set using PCA (via SVD).
    Returns:
        corners_world: (8,3) array, world-space corners in a consistent order
        center: (3,) world-space center of the OBB
        R: (3,3) rotation matrix, columns = OBB axes (right-handed, orthonormal)
        extents: (3,) box edge lengths along OBB axes (positive)
        lo, hi: (3,), (3,) min/max in OBB local coords (so that hi - lo = extents)
    Notes:
        - Stable to degeneracy; if rank < 3, falls back gracefully.
        - Axis signs are stabilized so the “positive” side tends to be the larger span.
    """
    P = np.asarray(points, dtype=np.float64)
    assert P.ndim == 2 and P.shape[1] == 3, "points must be (N,3)"
    N = P.shape[0]
    if N == 0:
        raise ValueError("No points for OBB")
    center = P.mean(axis=0)
    X = P - center

    # Degenerate quick-outs
    if N == 1:
        R = np.eye(3)
        extents = np.array([eps, eps, eps])
        lo = -0.5 * extents
        hi = 0.5 * extents
        corners_local = _corners_from_lo_hi(lo, hi)
        return _world_from_local(corners_local, center, R), center, R, extents, lo, hi

    U, S, VT = np.linalg.svd(X, full_matrices=False)  # X ≈ U * diag(S) * VT
    R = VT.T  # columns are principal axes
    order = np.argsort(-S)[:3]
    R = R[:, order]
    S = S[order]

    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0

    Y = X @ R

    lo = Y.min(axis=0)
    hi = Y.max(axis=0)

    for k in range(3):
        if abs(lo[k]) > abs(hi[k]):
            # Flip axis k
            R[:, k] *= -1.0
            lo[k], hi[k] = -hi[k], -lo[k]
    extents = np.maximum(hi - lo, eps)
    local_center = 0.5 * (lo + hi)
    center = (
        local_center @ R.T + center
    )  # same as original center, numerically consistent
    shift = local_center
    lo = lo - shift
    hi = hi - shift

    corners_local = _corners_from_lo_hi(lo, hi)
    corners_world = _world_from_local(corners_local, center, R)

    return corners_world, center, R, extents, lo, hi


def compute_yaw_obb(points: np.ndarray, eps: float = 1e-6):

    def _yaw_from_cov_xy(XY: np.ndarray) -> float:
        # XY: (N,2) centered points
        C = XY.T @ XY / max(XY.shape[0] - 1, 1)
        # principal dir is eigenvector of largest eigval
        vals, vecs = np.linalg.eigh(C)  # eigh: symmetric -> sorted ascending
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

    # yaw from XY covariance (ignore z for orientation)
    yaw = _yaw_from_cov_xy(X[:, :2])
    Rz = _R_from_yaw(yaw)

    # project to yaw frame and take AABB there -> extents & corners
    Y = X @ Rz
    lo = Y.min(axis=0)
    hi = Y.max(axis=0)
    extents = np.maximum(hi - lo, eps)

    # center box around its local center (no sign heuristics; prevents flicker)
    local_center = 0.5 * (lo + hi)
    lo -= local_center
    hi -= local_center
    center_world = center + local_center @ Rz.T

    corners_local = _corners_from_lo_hi(lo, hi)  # your existing function
    corners_world = _world_from_local(corners_local, center_world, Rz)

    return corners_world, center_world, Rz, extents, lo, hi, yaw


def _corners_from_lo_hi(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Return 8 corners in a consistent order from per-axis lo/hi in local coords."""
    # Order: (x,y,z) ∈ {lo,hi}^3 with a conventional pattern
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
    assert cluster_data.shape[1] == 5
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


def compute_clusters(scene: Scene) -> Scene:
    """
    Obtains clusters from a single scene. Returns a Scene object with set clusters
    """
    raw_clusters = _cluster_scene_dbscan(scene.points[:, :3], voxel_eps=2.0)
    cluster_labels = np.unique(raw_clusters)
    scene_clusters: list[Cluster] = []
    for c in cluster_labels:
        correct_mask = np.nonzero(raw_clusters == c)[0]
        if correct_mask.shape[0] <= 10:  # Remove small clusters
            continue
        cluster_points = scene.points[correct_mask, :]
        cluster_geo = _compute_cluster_geometry(cluster_points)
        cluster = Cluster(
            member_indices=correct_mask.tolist(), geometry=cluster_geo, label=c
        )

        scene_clusters.append(cluster)

    scene.scene_clusters = scene_clusters

    return scene
