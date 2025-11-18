from __future__ import annotations
from typing import Sequence, Dict, Optional, Tuple
import numpy as np
import open3d as o3d
import matplotlib as mpl

from lidar_types import Scene

# Fixed 12 cube edges for the ordering:
# 0:(l,l,l) 1:(h,l,l) 2:(l,h,l) 3:(l,l,h) 4:(h,h,l) 5:(h,l,h) 6:(l,h,h) 7:(h,h,h)
_CUBE_EDGES = np.array(
    [
        # bottom face (z = lo)
        [0, 1],
        [1, 4],
        [4, 2],
        [2, 0],
        # top face (z = hi)
        [3, 5],
        [5, 7],
        [7, 6],
        [6, 3],
        # verticals
        [0, 3],
        [1, 5],
        [2, 6],
        [4, 7],
    ],
    dtype=np.int32,
)


def lineset_from_ordered_corners(corners_world: np.ndarray) -> o3d.geometry.LineSet:
    """
    corners_world: (8,3) in the fixed order shown above.
    Returns a LineSet with all 12 edges.
    """
    assert corners_world.shape == (8, 3)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_world.astype(np.float64, copy=False))
    ls.lines = o3d.utility.Vector2iVector(_CUBE_EDGES)
    return ls


def bbox_diag_length_from_corners(corners: np.ndarray) -> float:
    return float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))


def bbox_diag_length_from_minmax(bb2: np.ndarray) -> float:
    return float(np.linalg.norm(bb2[1] - bb2[0]))


class ClusterBBoxViewer:
    """
    Simple Open3D legacy viewer:
      - Renders points
      - Renders all cluster bounding boxes, filtered by diag-length
      - ←/A prev, →/D next, B toggle boxes, R reset, Q/ESC quit
    """

    def __init__(
        self,
        scenes: list[Scene],
        entity_ids: list[np.ndarray] = [],
        min_diag: float = 0.1,
        max_diag: float = 1e6,
        point_size: float = 2.0,
        box_line_width: float = 1.5,
    ):
        if not scenes:
            raise ValueError("Provide at least one scene")
        self.scenes = scenes
        self.idx = 0
        self.min_diag = float(min_diag)
        self.max_diag = float(max_diag)
        self.show_boxes = True

        # Legacy visualizer for simplicity
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd = o3d.geometry.PointCloud()
        self._added = False

        self._scene_box_cache: Dict[int, list[Tuple[str, o3d.geometry.Geometry]]] = {}
        self._active_box_geoms: list[o3d.geometry.Geometry] = []

        # materials-ish (legacy)
        self._point_size = float(point_size)
        self._box_line_width = float(box_line_width)

        self.entity_ids = entity_ids

        self.mode = 1  # Track visualization, 1 if velocity field

    def _colors_from_labels(self, labels: Optional[np.ndarray], N: int) -> np.ndarray:
        if labels is None or len(labels) != N:
            return np.full((N, 3), 0.75, float)
        labels = labels.astype(int, copy=False)
        valid = labels >= 0
        # short palette
        base = np.array(
            [
                [0.121, 0.466, 0.705],
                [1.000, 0.498, 0.054],
                [0.172, 0.627, 0.172],
                [0.839, 0.152, 0.156],
                [0.580, 0.404, 0.741],
                [0.549, 0.337, 0.294],
                [0.890, 0.467, 0.761],
                [0.498, 0.498, 0.498],
                [0.737, 0.741, 0.133],
                [0.090, 0.745, 0.812],
            ],
            dtype=float,
        )
        K = int(labels[valid].max() + 1) if valid.any() else 1
        pal = base if K <= len(base) else np.vstack([base, base[: (K - len(base))]])
        cols = np.full((N, 3), 0.6, float)
        cols[valid] = pal[labels[valid] % len(pal)]
        return cols

    def _color_from_float(self, values: np.ndarray) -> np.ndarray:
        colormap = mpl.colormaps.get("plasma", None)
        if colormap is not None:
            return np.asarray(colormap(values)[:, :3])
        return np.full((values.shape[0], 3), 0.75, float)

    def _compute_boxes_for_scene(self, s_idx: int):
        if s_idx in self._scene_box_cache:
            return

        s = self.scenes[s_idx]
        boxes: list[Tuple[str, o3d.geometry.Geometry]] = []

        scene_clusters = getattr(s, "scene_clusters", None)
        if scene_clusters:
            for cl in scene_clusters:
                bb = getattr(cl.geometry, "bbox", None)
                ls = None
                if isinstance(bb, np.ndarray) and bb.shape == (8, 3):
                    diag = bbox_diag_length_from_corners(bb)
                    if self.min_diag <= diag <= self.max_diag:
                        ls = lineset_from_ordered_corners(bb)

                if ls is not None:
                    # lightly tint boxes; color by cluster label index
                    color = (
                        np.array(
                            [
                                (getattr(cl, "label", 0) * 37) % 255,
                                (getattr(cl, "label", 0) * 73) % 255,
                                (getattr(cl, "label", 0) * 19) % 255,
                            ]
                        )
                        / 255.0
                    ).tolist()
                    color = self._colors_from_labels(np.array([cl.label]), 1)[
                        0
                    ].tolist()
                    ls.colors = o3d.utility.Vector3dVector(
                        np.tile(color, (len(ls.lines), 1))
                    )
                    name = f"bbox_{s_idx}_{getattr(cl, 'label', 0)}"
                    boxes.append((name, ls))

        self._scene_box_cache[s_idx] = boxes

    def _add_current_scene_boxes(self):
        """Add boxes for self.idx and remember which ones were added."""
        self._active_box_geoms.clear()
        for _, geom in self._scene_box_cache.get(self.idx, []):
            self.vis.add_geometry(geom, reset_bounding_box=False)
            self._active_box_geoms.append(geom)

    def _remove_all_boxes(self):
        """Remove whatever boxes are currently in the window (regardless of scene)."""
        for geom in self._active_box_geoms:
            try:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            except Exception:
                pass
        self._active_box_geoms.clear()

    def _set_scene(self, idx: int):
        self.idx = max(0, min(idx, len(self.scenes) - 1))
        s = self.scenes[self.idx]

        pts_all = np.asarray(s.points[:, :3], dtype=np.float64)
        good = np.isfinite(pts_all).all(axis=1)
        pts = pts_all[good]
        self.pcd.points = o3d.utility.Vector3dVector(pts)

        # colors
        labels = getattr(s, "cluster_labels", None)
        labels = self.entity_ids[self.idx]

        print(s.velocity_field)
        velocities = getattr(s, "velocity_field", np.zeros((pts.shape[0], 3)))
        print(velocities)
        velocities = np.linalg.norm(velocities, axis=1)
        if self.mode == 0:
            cols = self._colors_from_labels(
                (
                    labels[good]
                    if (labels is not None and len(labels) == len(pts_all))
                    else None
                ),
                len(pts),
            )
        elif self.mode == 1 and velocities is not None:
            cols = self._color_from_float(velocities)
        else:
            cols = np.full((pts.shape[0], 3), 0.6, float)
        self.pcd.colors = o3d.utility.Vector3dVector(cols)

        if self._added:
            self.vis.update_geometry(self.pcd)
        else:
            self.vis.add_geometry(self.pcd, reset_bounding_box=True)
            self._added = True

        # adjust render options
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.02, 0.02, 0.025])
        opt.point_size = self._point_size
        try:
            opt.line_width = self._box_line_width  # available in newer legacy builds
        except AttributeError:
            pass

        # cache boxes for this scene (once)
        self._compute_boxes_for_scene(self.idx)

        # remove any old boxes, then (re)add according to toggle
        self._remove_all_boxes()
        if self.show_boxes:
            self._add_current_scene_boxes()

        # finally, refresh
        self.vis.update_renderer()

    # ---- key callbacks ----

    def _cb_next(self, vis):
        self._set_scene(self.idx + 1)
        return False

    def _cb_prev(self, vis):
        self._set_scene(self.idx - 1)
        return False

    def _cb_toggle_boxes(self, vis):
        self.show_boxes = not self.show_boxes
        if self.show_boxes:
            self._add_current_scene_boxes()
        else:
            self._remove_all_boxes()
        self.vis.update_renderer()
        return False

    def _cb_reset(self, vis):
        self.vis.reset_view_point(True)
        return False

    def _cb_quit(self, vis):
        self.vis.close()
        return False

    # ---- run ----

    def run(self, title: str = "Cluster BBoxes", w: int = 1280, h: int = 768):
        self.vis.create_window(title, width=w, height=h)
        self._set_scene(0)

        # keys
        self.vis.register_key_callback(262, self._cb_next)  # →
        self.vis.register_key_callback(ord("D"), self._cb_next)
        self.vis.register_key_callback(263, self._cb_prev)  # ←
        self.vis.register_key_callback(ord("A"), self._cb_prev)
        self.vis.register_key_callback(ord("B"), self._cb_toggle_boxes)  # toggle boxes
        self.vis.register_key_callback(ord("R"), self._cb_reset)
        self.vis.register_key_callback(256, self._cb_quit)  # ESC
        self.vis.register_key_callback(ord("Q"), self._cb_quit)

        print("[Controls] ←/A prev • →/D next • B boxes on/off • R reset • Q/ESC quit")
        try:
            self.vis.run()
        finally:
            self.vis.destroy_window()


# Convenience function
def view_cluster_bboxes(
    viewer: ClusterBBoxViewer,
    title: str = "Cluster BBoxes",
    w: int = 1280,
    h: int = 768,
):
    return viewer.run(title=title, w=w, h=h)
