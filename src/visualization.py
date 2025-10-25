from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import open3d as o3d
from lidar_types import Cluster, Scene

# ---- Minimal cluster-color viewer ----
class ClusterViewer:
    """Minimal Open3D viewer: just points, colored by cluster label.

    Keys: → / D = next  |  ← / A = prev  |  R = reset view  |  Q / ESC = quit
    """

    def __init__(self, scenes: Sequence[Scene]):
        if not scenes:
            raise ValueError("Provide at least one scene")
        self.scenes = list(scenes)
        self.idx = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd = o3d.geometry.PointCloud()
        self._added = False

    # --- colors ---
    @staticmethod
    def _palette(K: int) -> np.ndarray:
        # stable vivid palette; extend with HSV if needed
        base = np.array([
            [0.121, 0.466, 0.705], [1.000, 0.498, 0.054], [0.172, 0.627, 0.172], [0.839, 0.152, 0.156],
            [0.580, 0.404, 0.741], [0.549, 0.337, 0.294], [0.890, 0.467, 0.761], [0.498, 0.498, 0.498],
            [0.737, 0.741, 0.133], [0.090, 0.745, 0.812],
        ])
        if K <= len(base):
            return base[:K]
        extra = K - len(base)
        hsv = np.zeros((extra, 3))
        hsv[:, 0] = np.linspace(0, 1, extra, endpoint=False)
        hsv[:, 1] = 0.85
        hsv[:, 2] = 0.95
        return np.vstack([base, ClusterViewer._hsv_to_rgb(hsv)])

    @staticmethod
    def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        i = np.floor(h * 6).astype(int)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i_mod = i % 6
        rgb = np.empty((h.shape[0], 3))
        m = i_mod == 0
        rgb[m] = np.stack([v[m], t[m], p[m]], 1)
        m = i_mod == 1
        rgb[m] = np.stack([q[m], v[m], p[m]], 1)
        m = i_mod == 2
        rgb[m] = np.stack([p[m], v[m], t[m]], 1)
        m = i_mod == 3
        rgb[m] = np.stack([p[m], q[m], v[m]], 1)
        m = i_mod == 4
        rgb[m] = np.stack([t[m], p[m], v[m]], 1)
        m = i_mod == 5
        rgb[m] = np.stack([v[m], p[m], q[m]], 1)
        return rgb

    def _colors_from_labels(self, labels: Optional[np.ndarray]) -> np.ndarray:
        N = len(self.scenes[self.idx].points)
        if labels is None or len(labels) != N:
            return np.full((N, 3), 0.75, float)  # fallback grey
        labels = labels.astype(int)
        valid = labels >= 0
        K = int(labels[valid].max() + 1) if valid.any() else 1
        pal = self._palette(max(K, 8))
        colors = np.full((N, 3), 0.6, float)
        colors[valid] = pal[labels[valid] % len(pal)]
        return colors

    def _set_scene(self, idx: int):
        self.idx = max(0, min(idx, len(self.scenes) - 1))
        s = self.scenes[self.idx]
        self.pcd.points = o3d.utility.Vector3dVector(s.points[:, :3])
        self.pcd.colors = o3d.utility.Vector3dVector(self._colors_from_labels(s.cluster_labels))
        if self._added:
            self.vis.update_geometry(self.pcd)
            self.vis.update_renderer()

    def _cb_next(self, vis):
        self._set_scene(self.idx + 1)
        return False

    def _cb_prev(self, vis):
        self._set_scene(self.idx - 1)
        return False

    def _cb_reset(self, vis):
        self.vis.reset_view_point(True)
        return False

    def _cb_quit(self, vis):
        self.vis.close()
        return False

    # --- run ---
    def run(self, title: str = "Cluster Viewer", w: int = 1280, h: int = 768):
        self.vis.create_window(title, width=w, height=h)
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.02, 0.02, 0.025])
        opt.point_size = 2.0

        self._set_scene(0)
        self.vis.add_geometry(self.pcd, reset_bounding_box=True)
        self._added = True

        # keys
        self.vis.register_key_callback(262, self._cb_next)  # →
        self.vis.register_key_callback(ord('D'), self._cb_next)
        self.vis.register_key_callback(263, self._cb_prev)  # ←
        self.vis.register_key_callback(ord('A'), self._cb_prev)
        self.vis.register_key_callback(ord('R'), self._cb_reset)
        self.vis.register_key_callback(256, self._cb_quit)  # ESC
        self.vis.register_key_callback(ord('Q'), self._cb_quit)

        print("[Controls] ←/A prev • →/D next • R reset • Q/ESC quit")
        try:
            self.vis.run()
        finally:
            self.vis.destroy_window()


def view_clusters(scenes: Sequence[Scene]):
    """
    Runs visualizer for scenes
    """
    ClusterViewer(scenes).run()
