from pathlib import Path
import json
import numpy as np

from lidar_types import Sweep


def _load_lidar_bin(path: Path) -> np.ndarray:
    """Load one .pcd.bin LiDAR sweep from V2X-Sim (float32 XYZ + intensity)."""
    # Each point has x, y, z, intensity
    path = Path(path)
    assert path.suffix == ".bin", f"Unexpected format: {path.suffix}"
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape((-1, 5))


def _load_sample_metadata(metadata_path: Path) -> dict:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def load_sweeps(
    root, top_id, scene_id, sweep_ids: list[int], metadata_path: Path | None
) -> list[Sweep]:
    """
    load
    """
    if metadata_path is not None:
        raw_metadata = _load_sample_metadata(metadata_path)
        metadata_dict = {}
        for element in raw_metadata:
            metadata_dict[element.get("filename")] = element
    else:
        metadata_dict = None

    return list(
        map(lambda x: _load_sweep(root, top_id, scene_id, x, metadata_dict), sweep_ids)
    )


def _load_sweep(
    root: Path,
    top_id: int,
    scene_id: int,
    sweep_id: int,
    metadata_dict: dict | None = None,
):
    """Return (points) for one sweep id like 1, 2, 3..."""
    sweep_path = f"sweeps/LIDAR_TOP_id_{top_id}/scene_{scene_id}_{sweep_id:06d}.pcd.bin"
    bin_path = root / sweep_path
    pts = _load_lidar_bin(bin_path)

    # Finding the metadata
    metadata = metadata_dict.get(sweep_path) if metadata_dict is not None else None
    return Sweep(pts, metadata or {})
