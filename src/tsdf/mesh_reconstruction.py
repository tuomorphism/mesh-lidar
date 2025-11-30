from typing import Optional
import numpy as np
from skimage import measure

from lidar_types import TrackHistory, TrackSnapshot, Scene


class TSDFVolume:
    """
    Class for creating a mesh based on static tracks
    """

    def __init__(
        self,
        bounds_min,
        bounds_max,
        voxel_size: float = 0.15,
        trunc_dist: float = 0.30,
    ):
        """
        World-aligned dense TSDF.

        bounds_* : 3D world coordinates of the volume corners.
        voxel_size : size of one voxel edge in meters.
        trunc_dist : truncation distance for SDF in meters.
        """
        self.bounds_min = np.asarray(bounds_min, dtype=np.float32)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float32)
        self.voxel_size = float(voxel_size)
        self.trunc = float(trunc_dist)

        dims = np.ceil((self.bounds_max - self.bounds_min) / self.voxel_size).astype(
            int
        )
        self.dims = tuple(int(d) for d in dims)

        # TSDF initialized to 0 (“far outside surface”)
        self.tsdf = np.zeros(self.dims, dtype=np.float32)
        self.weights = np.zeros(self.dims, dtype=np.float32)

    def world_to_voxel(self, pts_world: np.ndarray) -> np.ndarray:
        """
        pts_world: (..., 3) array in world coordinates
        returns int voxel indices (..., 3)
        """
        return ((pts_world - self.bounds_min) / self.voxel_size).astype(np.int32)

    def integrate_points(self, pts_world: np.ndarray) -> None:
        """
        Naive point-based TSDF update:
        each point only updates the voxel it falls into.

        pts_world: (N,3)
        """
        if pts_world.size == 0:
            return
        for p in pts_world:
            idx = self.world_to_voxel(p)
            ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])

            if not (
                0 <= ix < self.dims[0]
                and 0 <= iy < self.dims[1]
                and 0 <= iz < self.dims[2]
            ):
                continue

            # center of that voxel in world coords
            center = self.bounds_min + (idx.astype(np.float32) + 0.5) * self.voxel_size

            # unsigned distance (simple MVP)
            dist = float(np.linalg.norm(p - center))
            u = np.clip(dist / self.trunc, 0.0, 1.0)
            sdf = 1.0 - u

            w_old = self.weights[ix, iy, iz]
            w_new = w_old + 1.0

            self.tsdf[ix, iy, iz] = (self.tsdf[ix, iy, iz] * w_old + sdf) / w_new
            self.weights[ix, iy, iz] = w_new

    def extract_mesh(self):
        """
        Returns vertices, faces, normals in *volume-local* coordinates.
        You can convert to world by adding bounds_min.
        """
        verts, faces, normals, _ = measure.marching_cubes(
            self.tsdf,
            level=0.5,
            spacing=(self.voxel_size, self.voxel_size, self.voxel_size),
        )
        # shift vertices into world coordinates
        verts_world = verts + self.bounds_min[None, :]
        return verts_world, faces, normals


def get_snapshot_for_scene(
    history: TrackHistory, scene_idx: int
) -> Optional[TrackSnapshot]:
    """
    Returns the snapshot of this track that corresponds to the given scene_idx,
    or None if the track was not present in that scene.
    """
    for snap in reversed(history.snapshots):
        if snap.scene_idx == scene_idx:
            return snap
        if snap.scene_idx < scene_idx:
            break
    return None


def collect_static_points_for_scene(
    scene: Scene,
    tracks: list[TrackHistory],
    scene_idx: int,
) -> np.ndarray:
    """
    Collects all points belonging to static tracks in the given scene.

    Returns:
        pts_static: (M, 3) array in world coords
    """
    pts_list = []

    for h in tracks:
        if not h.is_static:
            continue

        snap = get_snapshot_for_scene(h, scene_idx)
        if snap is None:
            continue

        pts_cluster = scene.points[snap.member_indices, :3]  # (K,3)
        pts_list.append(pts_cluster)

    if not pts_list:
        return np.empty((0, 3), dtype=np.float32)

    return np.concatenate(pts_list, axis=0).astype(np.float32)


def reconstruct_mesh(
    scenes: list[Scene], track_histories: list[TrackHistory]
) -> TSDFVolume:
    """
    Reconstruct a static TSDF volume from scenes + track histories.
    """

    static_points_all: list[np.ndarray] = []

    for scene_idx, scene in enumerate(scenes):
        static_points = collect_static_points_for_scene(
            scene=scene, tracks=track_histories, scene_idx=scene_idx
        )
        if static_points.size > 0:
            static_points_all.append(static_points)

    if not static_points_all:
        raise ValueError("No static points found in any scene.")

    all_pts = np.concatenate(static_points_all, axis=0)

    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)

    margin = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    bounds_min = (mins - margin).astype(np.float32)
    bounds_max = (maxs + margin).astype(np.float32)

    tsdf_volume = TSDFVolume(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        voxel_size=0.20,
        trunc_dist=1.0,
    )

    for static_points in static_points_all:
        tsdf_volume.integrate_points(static_points)

    return tsdf_volume
