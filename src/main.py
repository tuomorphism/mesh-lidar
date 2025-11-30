from pathlib import Path
from tqdm import tqdm

from visualization.scene_track_visualization import SceneAndTrackViewer
from scene_processing.loader import load_sequence_timesynced
from scene_processing.processor import process_sweeps
from tracking.tracking import Tracker, TrackingConf
from tsdf.classification_track import classify_static_dynamic, StaticDynamicConf
from tsdf.filter_tracks import filter_tracking_result
from fusion import fuse_sweeps_in_world


def main(
    dataset_designation: Path = Path("UrbanIng-V2X/dataset/20241126_0017_crossing1_00"),
):
    root = Path("./datasets") / dataset_designation
    N = 200
    sweep_data = load_sequence_timesynced(root, max_frames=N)
    timestamps = list(sweep_data.keys())

    print(f"Number of frames {len(timestamps)}.")

    combined_sweeps = []
    total_point_count = 0
    for ts in tqdm(timestamps[:N], desc="Fusing sweeps"):
        sweep_11 = sweep_data[ts]["crossing1_11_lidar"]
        sweep_12 = sweep_data[ts]["crossing1_12_lidar"]

        sweep_31 = sweep_data[ts]["crossing1_31_lidar"]
        sweep_32 = sweep_data[ts]["crossing1_32_lidar"]
        combined_sweep = fuse_sweeps_in_world([sweep_11, sweep_12, sweep_31, sweep_32])
        combined_sweeps.append(combined_sweep)

        total_point_count += combined_sweep.pts.shape[0]
    print(f"Total point count: {total_point_count} points")

    processed = process_sweeps(combined_sweeps[:N])

    print("Creating tracks")
    tracker_conf = TrackingConf()
    tracker = Tracker(tracker_conf)
    result = tracker.fit(processed)

    classification_conf = StaticDynamicConf()
    for history in tqdm(result.histories, desc="Classifying static/dynamic"):
        classify_static_dynamic(history, classification_conf, processed)

    result = filter_tracking_result(result, keep_static=True)

    viewer = SceneAndTrackViewer(processed, tracking_result=result, min_diag=0.1)
    viewer.run()


if __name__ == "__main__":
    main()
