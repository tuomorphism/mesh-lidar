from pathlib import Path
import open3d as o3d
import numpy as np


from visualization.visualization import ClusterBBoxViewer, view_cluster_bboxes
from scene_processing.loader import load_sequence_timesynced
from scene_processing.processor import process_sweeps
from tracking.tracking import Tracker, TrackingConf
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
    root = Path("./datasets/UrbanIng-V2X/dataset/20241126_0017_crossing1_00")
    N = 200
    sweep_data = load_sequence_timesynced(root)
    timestamps = list(sweep_data.keys())

    print(f"Number of frames {len(timestamps)}.")

    combined_sweeps = []
    total_point_count = 0
    for ts in timestamps[:N]:
        sweep_11 = sweep_data[ts]["crossing1_11_lidar"]
        sweep_12 = sweep_data[ts]["crossing1_12_lidar"]

        sweep_31 = sweep_data[ts]["crossing1_31_lidar"]
        sweep_32 = sweep_data[ts]["crossing1_32_lidar"]
        combined_sweep = fuse_sweeps_in_world([sweep_11, sweep_12, sweep_31, sweep_32])
        combined_sweeps.append(combined_sweep)

        total_point_count += combined_sweep.pts.shape[0]
        print(f"Total point count: {total_point_count} points")

    processed = process_sweeps(combined_sweeps[:N])

    tracker_conf = TrackingConf()
    tracker = Tracker(tracker_conf)
    result = tracker.apply(processed)

    viewer = ClusterBBoxViewer(processed, result.point_to_entity_per_scene or [])
    view_cluster_bboxes(viewer)


if __name__ == "__main__":
    main()
