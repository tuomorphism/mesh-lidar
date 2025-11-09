from pathlib import Path
from visualization import ClusterBBoxViewer, view_cluster_bboxes
from loader import load_sweeps
from processor import process_sweeps
from tracking import Tracker, TrackingConf


if __name__ == "__main__":
    root = Path("./data/lidar1/")
    N = 64
    scene_indices = [6 + i for i in range(N)]
    sweep_data = load_sweeps(
        root, 0, 1, scene_indices, Path("./data/meta/sample_data.json")
    )
    processed = process_sweeps(sweep_data[:N])

    tracker_conf = TrackingConf()
    tracker = Tracker(tracker_conf)
    result = tracker.apply(processed)

    viewer = ClusterBBoxViewer(processed, result.point_to_entity_per_scene or [])
    view_cluster_bboxes(viewer)
