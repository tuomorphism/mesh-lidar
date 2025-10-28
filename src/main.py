from pathlib import Path
from visualization import ClusterBBoxViewer, view_cluster_bboxes
from loader import load_sweep
from processor import process_sweeps
from tracking import Tracker, TrackingConf


if __name__ == "__main__":
    root = Path("./data/lidar1/")
    scene_indices = [6 + i for i in range(100 - 6 + 1)]
    sweeps = [load_sweep(root, 0, 1, idx) for idx in scene_indices]
    print(len(sweeps))
    processed = process_sweeps(sweeps[:100])

    viewer = ClusterBBoxViewer(processed)
    tracker_conf = TrackingConf()
    tracker = Tracker(tracker_conf, viewer)
    tracker.apply(processed)

    view_cluster_bboxes(viewer)
