import numpy as np
import open3d as o3d

# --- utilities to go between SE(3) state and visuals ---


def _ordered_corners_from_pose_size(T: np.ndarray, size: np.ndarray) -> np.ndarray:
    """
    Build the 8 bbox corners in the *fixed* order your viewer uses:
    0:(l,l,l) 1:(h,l,l) 2:(l,h,l) 3:(l,l,h) 4:(h,h,l) 5:(h,l,h) 6:(l,h,h) 7:(h,h,h)
    where l/h are local min/max along the object's principal axes.
    """
    R = T[:3, :3]
    c = T[:3, 3]
    hx, hy, hz = 0.5 * size
    # local min/max
    lo = np.array([-hx, -hy, -hz], float)
    hi = np.array([+hx, +hy, +hz], float)

    # follow the exact order your LineSet expects
    corners_local = np.array(
        [
            [lo[0], lo[1], lo[2]],  # 0
            [hi[0], lo[1], lo[2]],  # 1
            [lo[0], hi[1], lo[2]],  # 2
            [lo[0], lo[1], hi[2]],  # 3
            [hi[0], hi[1], lo[2]],  # 4
            [hi[0], lo[1], hi[2]],  # 5
            [lo[0], hi[1], hi[2]],  # 6
            [hi[0], hi[1], hi[2]],  # 7
        ],
        dtype=float,
    )

    return (corners_local @ R.T) + c  # (8,3)


def _centroid_from_T(T: np.ndarray) -> np.ndarray:
    return T[:3, 3].astype(float, copy=False)


def _centroid_from_cluster(cl) -> np.ndarray:
    return np.asarray(cl.geometry.centroid, dtype=float).reshape(3)


def _track_color(track_id: int) -> np.ndarray:
    # deterministic pleasant palette from id
    np.random.seed((track_id * 2654435761) % (2**32))
    base = np.array([0.12, 0.48, 0.82])  # bluish
    jitter = np.random.rand(3) * 0.5
    col = 0.5 * base + 0.5 * jitter
    return np.clip(col, 0.15, 0.95)


# --- main overlay object ---


class MatchOverlay:
    """
    Stores per-scene overlay geometries:
      - predicted bboxes (from SE(3) predicted T and size)
      - lines from predicted centroids to measured centroids
    """

    def __init__(self, lineset_from_ordered_corners, cube_edges):
        self.lineset_from_ordered_corners = lineset_from_ordered_corners
        self._cube_edges = cube_edges
        self._scene_overlays = {}  # scene_index -> list[Geometry]
        self._active = []

    def clear_active(self, vis):
        for g in self._active:
            try:
                vis.remove_geometry(g, reset_bounding_box=False)
            except Exception:
                pass
        self._active.clear()

    def add_active(self, vis, scene_index: int):
        geoms = self._scene_overlays.get(scene_index, [])
        for g in geoms:
            vis.add_geometry(g, reset_bounding_box=False)
            self._active.append(g)

    def add_matches(
        self,
        prev_scene_index: int,
        next_scene_index: int,
        assignments: list[tuple[int, int]],
        predicted_poses: list[
            np.ndarray
        ],  # len == #tracks (same order as assignments' row idx)
        track_ids: list[int],  # len == #tracks
        next_clusters: list,
        predicted_sizes: list[np.ndarray] | None = None,  # optional len==#tracks
        thin_pred_lines: bool = True,
    ):
        """
        Precompute overlay for next_scene_index using the given assignment
        and tracker predictions.
        """
        overlays = []

        # build one merged LineSet for centroid->centroid lines (efficient)
        line_pts = []
        line_idx = []
        line_cols = []

        for pair in assignments:
            i_track, j_cl = pair
            T_pred = predicted_poses[i_track]
            col = _track_color(track_ids[i_track])

            # predicted bbox (if size known)
            if predicted_sizes is not None and predicted_sizes[i_track] is not None:
                corners_pred = _ordered_corners_from_pose_size(
                    T_pred, predicted_sizes[i_track]
                )
                ls_pred = self.lineset_from_ordered_corners(corners_pred)
                # thinner / dimmer so it doesn't fight with measured boxes
                ls_pred.colors = o3d.utility.Vector3dVector(
                    np.tile(
                        col * 0.7 + 0.3 * np.array([0.9, 0.9, 0.9]),
                        (len(ls_pred.lines), 1),
                    )
                )
                overlays.append(ls_pred)

            # line from predicted centroid to measured centroid
            p0 = _centroid_from_T(T_pred)
            p1 = _centroid_from_cluster(next_clusters[j_cl])
            base_idx = len(line_pts)
            line_pts.extend([p0, p1])
            line_idx.append([base_idx, base_idx + 1])
            line_cols.append(col)

        if line_idx:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(np.asarray(line_pts, float))
            ls.lines = o3d.utility.Vector2iVector(np.asarray(line_idx, np.int32))
            ls.colors = o3d.utility.Vector3dVector(np.asarray(line_cols, float))
            overlays.append(ls)

        self._scene_overlays[next_scene_index] = overlays
