from pathlib import Path
from visualization.visualization import ClusterBBoxViewer, view_cluster_bboxes
from scene_processing.loader import load_sequence_timesynced
from scene_processing.processor import process_sweeps
from tracking.tracking import Tracker, TrackingConf

import numpy as np
import open3d as o3d
from lidar_types import Sweep
from fusion import fuse_sweeps_in_world


def visualize_sweep(
    sweep: Sweep,
    color_mode: str = "height",
):
    """
    Visualize a single LiDAR or radar sweep using Open3D.

    Args:
        sweep         : Sweep object containing pts and metadata
        voxel_size    : Downsample size (meters)
        color_mode    : 'height' or 'intensity'
    """

    pts = sweep.pts

    if pts.shape[1] < 3:
        raise ValueError("Sweep must contain at least XYZ columns.")

    xyz = pts[:, :3]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # ----- Color handling -----
    if color_mode == "height":
        z = xyz[:, 2]
        z_min, z_max = np.min(z), np.max(z)
        z_norm = (z - z_min) / max(z_max - z_min, 1e-6)
        colors = np.stack([z_norm, 1 - z_norm, 0.5 * np.ones_like(z_norm)], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    elif color_mode == "intensity" and pts.shape[1] >= 4:
        intensity = pts[:, 3]
        i = intensity - intensity.min()
        if i.max() > 0:
            i /= i.max()
        colors = np.stack([i, i, i], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    else:
        # Default color: light blue
        pcd.paint_uniform_color([0.2, 0.5, 1.0])

    # ----- Visualize -----
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Sweep visualization: {sweep.metadata.get('sensor', '')}",
        point_show_normal=False,
        width=1280,
        height=720,
    )


def main():
    # datasets/UrbanIng-V2X/dataset/20241126_0017_crossing1_00/timesync_info.csv
    root = Path("./datasets/UrbanIng-V2X/dataset/20241126_0017_crossing1_00")
    N = 128
    sweep_data = load_sequence_timesynced(root)

    # print(sweep_data)

    timestamps = list(sweep_data.keys())

    combined_sweeps = []
    for ts in timestamps:
        sweep_1 = sweep_data[ts]["crossing1_11_lidar"]
        sweep_2 = sweep_data[ts]["crossing1_12_lidar"]
        combined_sweep = fuse_sweeps_in_world([sweep_1, sweep_2])
        combined_sweeps.append(combined_sweep)

    # visualize_sweep(combined_sweep)

    processed = process_sweeps(combined_sweeps[:N])

    tracker_conf = TrackingConf()
    tracker = Tracker(tracker_conf)
    result = tracker.apply(processed)

    viewer = ClusterBBoxViewer(processed, result.point_to_entity_per_scene or [])
    view_cluster_bboxes(viewer)


if __name__ == "__main__":
    main()
