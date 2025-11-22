import csv
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional

from lidar_types import Sweep


def _load_lidar_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    x_values = data["x"]
    y_values = data["y"]
    z_values = data["z"]
    int_values = data["intensity"]
    return np.column_stack([x_values, y_values, z_values, int_values])


def load_timesync_matrix(path: Path) -> List[List[str]]:
    with open(path, "r") as f:
        reader = csv.reader(f)
        return [row for row in reader]


def load_calibration(path: Path) -> Dict[str, dict]:
    """
    loads calibration jsonfile from path
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_extrinsic_matrix(block: dict) -> Optional[np.ndarray]:
    """Extract the first 4×4 extrinsic matrix in the block."""
    if "extrinsics" not in block:
        return None
    for _, value in block["extrinsics"].items():
        mat = np.array(value)
        if mat.shape == (4, 4):
            return mat
    return None


def load_sequence_timesynced(
    sequence_root: Path,
    lidar_suffix: str = "_lidar",
    max_frames: Optional[int] = None,
) -> Dict[int, Dict[str, Sweep]]:

    matrix = load_timesync_matrix(sequence_root / "timesync_info.csv")

    sensors = [row[0] for row in matrix[1:]]
    timestamp_row = matrix[1]
    timestamps = [int(x) for x in timestamp_row[1:]]

    calib = load_calibration(sequence_root / "calibration.json")
    timeline: Dict[int, Dict[str, Sweep]] = {}

    num_loaded = 0

    for j, ts in enumerate(timestamps):
        if max_frames is not None and num_loaded >= max_frames:
            break

        frame_dict: Dict[str, Sweep] = {}

        for i, sensor_name in enumerate(sensors, start=1):
            filename = matrix[i][j + 1]
            if not filename:
                continue

            if not (
                sensor_name.endswith(lidar_suffix)
                and sensor_name.startswith("crossing")
            ):
                continue

            sensor_dir = sequence_root / sensor_name
            file_path = sensor_dir / filename
            if not file_path.exists():
                continue

            pts = _load_lidar_npz(file_path)

            metadata = {
                "timestamp_ms": ts,
                "timestamp": ts,
                "sensor": sensor_name,
                "filename": filename,
                "path": str(file_path),
            }

            if sensor_name in calib:
                cal_block = calib[sensor_name]
                T = _extract_extrinsic_matrix(cal_block)
                metadata["calibration"] = cal_block
                if T is not None:
                    metadata["extrinsics"] = {
                        "T_sensor_in_parent": T,
                        "parent_T_sensor": np.linalg.inv(T),
                        "type": list(cal_block["extrinsics"].keys())[0],
                    }

            frame_dict[sensor_name] = Sweep(pts, metadata)

        timeline[ts] = frame_dict
        num_loaded += 1

    return timeline
