from pathlib import Path
import numpy as np


def _load_lidar_bin(path: Path) -> np.ndarray:
    """Load one .pcd.bin LiDAR sweep from V2X-Sim (float32 XYZ + intensity)."""
    # Each point has x, y, z, intensity
    path = Path(path)
    assert path.suffix == ".bin", f"Unexpected format: {path.suffix}"
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape((-1, 5))


def load_sweep(root: Path, top_id: int, scene_id: int, sweep_id: int):
    """Return (points) for one sweep id like 1, 2, 3..."""
    bin_path = (
        root
        / "sweeps"
        / f"LIDAR_TOP_id_{top_id}/scene_{scene_id}_{sweep_id:06d}.pcd.bin"
    )
    pts = _load_lidar_bin(bin_path)
    return pts
