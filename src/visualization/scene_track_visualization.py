from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import open3d as o3d
import matplotlib as mpl

from lidar_types import Scene, TrackHistory

# ---------------------------------------------------------------------
# Common geometry helpers
# ---------------------------------------------------------------------

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


def ellipse_lineset_from_cov_xy(
    mean_xy: np.ndarray,
    cov_xy: np.ndarray,
    z: float,
    n_points: int = 64,
    n_std: float = 2.0,
) -> Optional[o3d.geometry.LineSet]:
    """
    Build a 2D covariance ellipse in the XY plane at height z as a LineSet.

    mean_xy: (2,)
    cov_xy: (2,2)
    """
    if cov_xy.shape != (2, 2):
        return None

    try:
        vals, vecs = np.linalg.eigh(cov_xy)
    except np.linalg.LinAlgError:
        return None

    # Guard against negative eigenvalues due to num. issues
    vals = np.maximum(vals, 1e-9)
    radii = n_std * np.sqrt(vals)  # scale by n_std standard deviations

    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2, N)
    # Map unit circle -> ellipse via eigenvectors and radii
    ellipse_xy = (vecs @ (radii[:, None] * circle)).T + mean_xy[None, :]  # (N,2)

    # lift to 3D
    pts = np.zeros((n_points, 3), dtype=float)
    pts[:, 0:2] = ellipse_xy
    pts[:, 2] = z

    # lines connecting in a loop
    lines = [[i, (i + 1) % n_points] for i in range(n_points)]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    return ls


# ---------------------------------------------------------------------
# Scene & track + mesh viewer
# ---------------------------------------------------------------------


class SceneAndTrackViewer:
    """
    Viewer for:
      - LiDAR scenes
      - Cluster bounding boxes
      - Multi-object tracks (trajectories)
      - EKF uncertainty ellipses (position covariance)
      - Optional static TSDF mesh of the environment

    Modes:
        1: Bounding boxes + entity-colored points
        2: Track trajectories
        3: Uncertainty ellipses
        4: Bounding boxes + tracks + uncertainty
        5: Velocity-colored points
        6: Intensity-colored points

    Extra:
        M   : toggle static mesh on/off

    Controls:
        ←/A : previous frame
        →/D : next frame
        1-6 : change visualization mode
        B   : toggle bounding boxes on/off (where relevant)
        M   : toggle static mesh on/off
        R   : reset viewpoint
        Q/ESC: quit
    """

    MODE_BBOX = 0
    MODE_TRACKS = 1
    MODE_UNCERT = 2
    MODE_BBOX_TRACKS = 3
    MODE_VELOCITY = 4
    MODE_INTENSITY = 5

    def __init__(
        self,
        scenes: List[Scene],
        tracking_result,
        min_diag: float = 0.1,
        max_diag: float = 1e6,
        point_size: float = 2.0,
        box_line_width: float = 1.5,
        static_mesh: Optional[o3d.geometry.TriangleMesh] = None,
        static_mesh_color: Optional[np.ndarray] = None,
    ):
        if not scenes:
            raise ValueError("Provide at least one scene")

        self.scenes: List[Scene] = scenes
        self.tracking_result = tracking_result
        self.histories: List[TrackHistory] = getattr(tracking_result, "histories", [])
        self.point_to_entity_per_scene: Optional[List[np.ndarray]] = getattr(
            tracking_result, "point_to_entity_per_scene", None
        )

        self.idx = 0
        self.min_diag = float(min_diag)
        self.max_diag = float(max_diag)
        self.show_boxes = True

        # --- static mesh handling -------------------------------------------
        self.static_mesh: Optional[o3d.geometry.TriangleMesh] = static_mesh
        self.show_mesh: bool = static_mesh is not None
        self._added_mesh: bool = False

        if self.static_mesh is not None:
            # Make a shallow copy so we can style it without touching original
            self.static_mesh = o3d.geometry.TriangleMesh(self.static_mesh)
            if not self.static_mesh.has_vertex_normals():
                self.static_mesh.compute_vertex_normals()

            if static_mesh_color is None:
                # slightly warm grey
                static_mesh_color = np.array([0.8, 0.8, 0.82], dtype=float)

            self.static_mesh.paint_uniform_color(static_mesh_color.astype(float))

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd = o3d.geometry.PointCloud()
        self._added_pcd = False

        # caches + active overlays
        self._scene_box_cache: Dict[int, List[o3d.geometry.LineSet]] = {}
        self._scene_cov_cache: Dict[int, List[o3d.geometry.LineSet]] = {}
        self._track_line_geoms: List[o3d.geometry.LineSet] = []

        self._active_box_geoms: List[o3d.geometry.Geometry] = []
        self._active_track_geoms: List[o3d.geometry.Geometry] = []
        self._active_cov_geoms: List[o3d.geometry.Geometry] = []

        # style
        self._point_size = float(point_size)
        self._box_line_width = float(box_line_width)

        # Start with combined bbox + tracks mode
        self.mode = self.MODE_BBOX_TRACKS

        # Precompute track trajectory line sets once
        self._build_track_lines()

    # ------------------------------------------------------------------
    # Color helpers
    # ------------------------------------------------------------------

    def _colors_from_labels(self, labels: Optional[np.ndarray], N: int) -> np.ndarray:
        """
        Map integer labels (e.g., entity IDs) to colors.
        """
        if labels is None or len(labels) != N:
            return np.full((N, 3), 0.75, float)
        labels = labels.astype(int, copy=False)
        valid = labels >= 0

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

    # ------------------------------------------------------------------
    # Box caching
    # ------------------------------------------------------------------

    def _compute_boxes_for_scene(self, s_idx: int):
        if s_idx in self._scene_box_cache:
            return

        s = self.scenes[s_idx]
        boxes: List[o3d.geometry.LineSet] = []

        scene_clusters = getattr(s, "scene_clusters", None)
        if scene_clusters:
            for cl in scene_clusters:
                bb = getattr(cl.geometry, "bbox", None)
                if not (isinstance(bb, np.ndarray) and bb.shape == (8, 3)):
                    continue

                diag = bbox_diag_length_from_corners(bb)
                if not (self.min_diag <= diag <= self.max_diag):
                    continue

                ls = lineset_from_ordered_corners(bb)

                # Color boxes by entity/cluster label if available
                label = getattr(cl, "label", 0)
                color = self._colors_from_labels(np.array([label]), 1)[0].tolist()
                ls.colors = o3d.utility.Vector3dVector(
                    np.tile(color, (len(ls.lines), 1))
                )
                boxes.append(ls)

        self._scene_box_cache[s_idx] = boxes

    def _add_current_scene_boxes(self):
        """Add boxes for self.idx and remember which ones were added."""
        self._compute_boxes_for_scene(self.idx)
        self._active_box_geoms.clear()
        for geom in self._scene_box_cache.get(self.idx, []):
            self.vis.add_geometry(geom, reset_bounding_box=False)
            self._active_box_geoms.append(geom)

    # ------------------------------------------------------------------
    # Track line building
    # ------------------------------------------------------------------

    def _build_track_lines(self):
        """
        Build LineSets for full trajectories of each track (across all frames).
        """
        self._track_line_geoms.clear()
        if not self.histories:
            return

        for hist in self.histories:
            snaps = hist.snapshots
            if len(snaps) < 2:
                continue

            # ensure sorted by time
            snaps = sorted(snaps, key=lambda s: (s.scene_idx, s.timestamp))
            pts = np.stack([snap.T_w[:3, 3] for snap in snaps], axis=0)  # (N,3)

            lines = np.array([[i, i + 1] for i in range(len(pts) - 1)], dtype=np.int32)

            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            ls.lines = o3d.utility.Vector2iVector(lines)

            # color by entity_id
            color = self._colors_from_labels(np.array([hist.entity_id]), 1)[0].tolist()
            ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))

            self._track_line_geoms.append(ls)

    def _add_track_lines(self):
        """
        Add precomputed track trajectories to the scene.
        """
        self._active_track_geoms.clear()
        for geom in self._track_line_geoms:
            self.vis.add_geometry(geom, reset_bounding_box=False)
            self._active_track_geoms.append(geom)

    # ------------------------------------------------------------------
    # Covariance (uncertainty) ellipses
    # ------------------------------------------------------------------

    def _compute_cov_for_scene(self, s_idx: int):
        if s_idx in self._scene_cov_cache:
            return

        geoms: List[o3d.geometry.LineSet] = []

        for hist in self.histories:
            for snap in hist.snapshots:
                if snap.scene_idx != s_idx:
                    continue

                # We assume TrackSnapshot.P is (5,5)
                P = getattr(snap, "P", None)
                if P is None or P.shape != (5, 5):
                    continue

                cov_xy = P[0:2, 0:2]
                mean_xy = snap.T_w[:2, 3]
                z = float(snap.T_w[2, 3])

                ls = ellipse_lineset_from_cov_xy(mean_xy, cov_xy, z)
                if ls is None:
                    continue

                color = self._colors_from_labels(np.array([hist.entity_id]), 1)[
                    0
                ].tolist()
                # lighten slightly
                color = [0.8 * c + 0.2 for c in color]
                ls.colors = o3d.utility.Vector3dVector(
                    np.tile(color, (len(ls.lines), 1))
                )

                geoms.append(ls)

        self._scene_cov_cache[s_idx] = geoms

    def _add_cov_for_scene(self, s_idx: int):
        self._compute_cov_for_scene(s_idx)
        self._active_cov_geoms.clear()
        for geom in self._scene_cov_cache.get(s_idx, []):
            self.vis.add_geometry(geom, reset_bounding_box=False)
            self._active_cov_geoms.append(geom)

    # ------------------------------------------------------------------
    # Overlay management
    # ------------------------------------------------------------------

    def _remove_all_overlays(self):
        """
        Remove all overlay geometries (boxes, track lines, cov ellipses).

        Note: does NOT touch the static mesh.
        """
        for geom in (
            self._active_box_geoms + self._active_track_geoms + self._active_cov_geoms
        ):
            try:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            except Exception:
                pass

        self._active_box_geoms.clear()
        self._active_track_geoms.clear()
        self._active_cov_geoms.clear()

    # ------------------------------------------------------------------
    # Static mesh management
    # ------------------------------------------------------------------

    def _update_mesh_in_vis(self):
        """
        Ensure the static mesh is in the visualizer if show_mesh is True,
        or removed otherwise.
        """
        if self.static_mesh is None:
            return

        if self.show_mesh and not self._added_mesh:
            # Add mesh (once); keep bounding box as is to avoid jumps
            self.vis.add_geometry(self.static_mesh, reset_bounding_box=False)
            self._added_mesh = True
        elif not self.show_mesh and self._added_mesh:
            try:
                self.vis.remove_geometry(self.static_mesh, reset_bounding_box=False)
            except Exception:
                pass
            self._added_mesh = False

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _set_scene(self, idx: int):
        self.idx = max(0, min(idx, len(self.scenes) - 1))
        s = self.scenes[self.idx]

        pts_all = np.asarray(s.points[:, :3], dtype=np.float64)
        good = np.isfinite(pts_all).all(axis=1)
        pts = pts_all[good]
        self.pcd.points = o3d.utility.Vector3dVector(pts)

        # ---- choose colors ---------------------------------------------------
        velocities = getattr(s, "velocity_field", None)
        if velocities is not None:
            velocities = np.asarray(velocities, dtype=float)
            if velocities.shape[0] == len(pts_all):
                vmag = np.linalg.norm(velocities, axis=1)
                vmag = (vmag - vmag.min()) / (vmag.max() - vmag.min() + 1e-9)
            else:
                vmag = None
        else:
            vmag = None

        # entity ID per point (from tracker)
        if self.point_to_entity_per_scene is not None and self.idx < len(
            self.point_to_entity_per_scene
        ):
            labels_all = self.point_to_entity_per_scene[self.idx]
        else:
            labels_all = None

        if self.mode in (
            self.MODE_BBOX,
            self.MODE_TRACKS,
            self.MODE_UNCERT,
            self.MODE_BBOX_TRACKS,
        ):
            # entity-colored points
            labels = (
                labels_all[good]
                if (labels_all is not None and len(labels_all) == len(pts_all))
                else None
            )
            cols = self._colors_from_labels(labels, len(pts))
        elif self.mode == self.MODE_VELOCITY and vmag is not None:
            cols = self._color_from_float(vmag[good])
        elif self.mode == self.MODE_INTENSITY and s.points.shape[1] >= 4:
            intensity = s.points[:, 3]
            i = intensity - intensity.min()
            if i.max() > 0:
                i /= i.max()
            cols_full = np.stack([i, 1 - i, 0.5 * np.ones_like(i)], axis=1)
            cols = cols_full[good]
        else:
            cols = np.full((pts.shape[0], 3), 0.6, float)

        self.pcd.colors = o3d.utility.Vector3dVector(cols)

        # ---- add/update geometry --------------------------------------------
        if self._added_pcd:
            self.vis.update_geometry(self.pcd)
        else:
            # First time: reset bounding box to fit points + mesh
            self.vis.add_geometry(self.pcd, reset_bounding_box=True)
            self._added_pcd = True

        # always keep mesh consistent with show_mesh flag
        self._update_mesh_in_vis()

        # render options
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.02, 0.02, 0.025])
        opt.point_size = self._point_size
        try:
            opt.line_width = self._box_line_width
        except AttributeError:
            pass

        # remove overlays, then add according to current mode
        self._remove_all_overlays()

        if self.mode in (self.MODE_BBOX, self.MODE_BBOX_TRACKS):
            if self.show_boxes:
                self._add_current_scene_boxes()

        if self.mode in (self.MODE_TRACKS, self.MODE_BBOX_TRACKS):
            self._add_track_lines()

        if self.mode in (self.MODE_UNCERT, self.MODE_BBOX_TRACKS):
            self._add_cov_for_scene(self.idx)

        self.vis.update_renderer()

    # ------------------------------------------------------------------
    # Key callbacks
    # ------------------------------------------------------------------

    def _cb_next(self, vis):
        self._set_scene(self.idx + 1)
        return False

    def _cb_prev(self, vis):
        self._set_scene(self.idx - 1)
        return False

    def _cb_toggle_boxes(self, vis):
        self.show_boxes = not self.show_boxes
        self._set_scene(self.idx)
        return False

    def _cb_toggle_mesh(self, vis):
        """
        Toggle static mesh visibility.
        """
        self.show_mesh = not self.show_mesh
        self._update_mesh_in_vis()
        self.vis.update_renderer()
        print(f"[Mesh] {'ON' if self.show_mesh else 'OFF'}")
        return False

    def _cb_reset(self, vis):
        self.vis.reset_view_point(True)
        return False

    def _cb_quit(self, vis):
        self.vis.close()
        return False

    def _cb_mode_bbox(self, vis):
        self.mode = self.MODE_BBOX
        self._set_scene(self.idx)
        print("[Mode] Bounding boxes + entity-colored points")
        return False

    def _cb_mode_tracks(self, vis):
        self.mode = self.MODE_TRACKS
        self._set_scene(self.idx)
        print("[Mode] Track trajectories")
        return False

    def _cb_mode_uncert(self, vis):
        self.mode = self.MODE_UNCERT
        self._set_scene(self.idx)
        print("[Mode] Uncertainty ellipses")
        return False

    def _cb_mode_bbox_tracks(self, vis):
        self.mode = self.MODE_BBOX_TRACKS
        self._set_scene(self.idx)
        print("[Mode] BBoxes + tracks + uncertainty")
        return False

    def _cb_mode_velocity(self, vis):
        self.mode = self.MODE_VELOCITY
        self._set_scene(self.idx)
        print("[Mode] Velocity-colored points")
        return False

    def _cb_mode_intensity(self, vis):
        self.mode = self.MODE_INTENSITY
        self._set_scene(self.idx)
        print("[Mode] Intensity-colored points")
        return False

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, title: str = "Scene & Tracks + Mesh", w: int = 1280, h: int = 768):
        self.vis.create_window(title, width=w, height=h)
        self._set_scene(0)

        # navigation
        self.vis.register_key_callback(262, self._cb_next)  # →
        self.vis.register_key_callback(ord("D"), self._cb_next)
        self.vis.register_key_callback(263, self._cb_prev)  # ←
        self.vis.register_key_callback(ord("A"), self._cb_prev)

        # overlays + modes
        self.vis.register_key_callback(ord("B"), self._cb_toggle_boxes)
        self.vis.register_key_callback(ord("M"), self._cb_toggle_mesh)
        self.vis.register_key_callback(ord("R"), self._cb_reset)
        self.vis.register_key_callback(256, self._cb_quit)  # ESC
        self.vis.register_key_callback(ord("Q"), self._cb_quit)

        # modes 1–6
        self.vis.register_key_callback(ord("1"), self._cb_mode_bbox)
        self.vis.register_key_callback(ord("2"), self._cb_mode_tracks)
        self.vis.register_key_callback(ord("3"), self._cb_mode_uncert)
        self.vis.register_key_callback(ord("4"), self._cb_mode_bbox_tracks)
        self.vis.register_key_callback(ord("5"), self._cb_mode_velocity)
        self.vis.register_key_callback(ord("6"), self._cb_mode_intensity)

        print(
            "[Controls] ←/A prev • →/D next • 1–6 modes • "
            "B boxes on/off • M mesh on/off • R reset • Q/ESC quit"
        )
        try:
            self.vis.run()
        finally:
            self.vis.destroy_window()


def view_scene_and_tracks(
    viewer: SceneAndTrackViewer,
    title: str = "Scene & Tracks + Mesh",
    w: int = 1280,
    h: int = 768,
):
    return viewer.run(title=title, w=w, h=h)
