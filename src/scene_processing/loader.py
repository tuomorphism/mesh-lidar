from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Optional

from lidar_types import Sweep


def _load_lidar_npz(path: Path) -> np.ndarray:
    """
    Load one UrbanIng-V2X LiDAR npz file.
    Assumes it contains either:
        - points under key 'points'
        - or the first array in the archive
    """
    data = np.load(path)
    if "points" in data:
        pts = data["points"]
    else:
        # Fallback: first array
        pts = next(iter(data.values()))
    return pts


def _load_sample_metadata(metadata_path: Path) -> Dict[str, dict]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {item["filename"]: item for item in raw}


def _get_lidar_folders(sequence_path: Path) -> List[Path]:
    """
    Returns all LiDAR sensor folders inside a sequence:
    e.g. lidar_01, lidar_02, ...
    """
    return sorted(
        [
            p
            for p in sequence_path.iterdir()
            if p.is_dir() and p.name.startswith("lidar")
        ]
    )


def _get_lidar_files(lidar_dir: Path) -> List[Path]:
    """
    Returns all .npz files in time order.
    """
    files = sorted(lidar_dir.glob("*.npz"))
    return files


def _load_sweep(
    file_path: Path,
    metadata_dict: Optional[Dict[str, dict]] = None,
) -> Sweep:
    pts = _load_lidar_npz(file_path)

    # Relative path key for metadata lookup, same as V2X-Sim loader style
    key = str(file_path.name)
    metadata = metadata_dict.get(key) if metadata_dict is not None else {}

    return Sweep(pts, metadata)


def load_sequence(
    root: Path,
    sequence_id: str,
    metadata_path: Optional[Path] = None,
) -> Dict[str, List[Sweep]]:
    """
    Load all sweeps from a sequence, for all lidar sensors.

    Returns dict:
        {
            "lidar_01": [Sweep, Sweep, ...],
            "lidar_02": [Sweep, Sweep, ...],
            ...
        }
    """
    seq_path = root / sequence_id
    assert seq_path.exists(), f"Sequence not found: {seq_path}"

    # Metadata optional
    metadata_dict = (
        _load_sample_metadata(metadata_path) if metadata_path is not None else None
    )

    lidar_folders = _get_lidar_folders(seq_path)

    output = {}

    for lidar_dir in lidar_folders:
        files = _get_lidar_files(lidar_dir)
        sweeps = [_load_sweep(f, metadata_dict) for f in files]

        output[lidar_dir.name] = sweeps

    return output
