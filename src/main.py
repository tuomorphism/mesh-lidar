from pathlib import Path
from visualization import view_cluster_bboxes
from loader import load_sweep
from processor import process_sweeps

if __name__ == '__main__':
    root = Path("./data/lidar1/")
    scene_indices = [6 + i for i in range(30 - 6 + 1)]
    sweeps = [load_sweep(root, 0, 1, idx) for idx in scene_indices]

    processed = process_sweeps(sweeps[:60])

    view_cluster_bboxes(processed)