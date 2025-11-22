from __future__ import annotations

from typing import Any, Dict, List, Sequence
import copy
import numpy as np
from lidar_types import Sweep


def _get_T_world_from_sensor(metadata: Dict[str, Any]) -> np.ndarray:
    """
    Extract a homogeneous transform T_world_from_sensor (4x4) from sweep metadata.

    Expected layouts (any of these are fine):
      - metadata['extrinsics']['T_sensor_in_parent'] : np.ndarray (4,4)
        where 'parent' is the world frame in this dataset.
      - metadata['calibration']['extrinsics']['gTl'] : list/list-like (4,4)
        where gTl = global/world <- lidar.

    Raises:
        KeyError if no suitable transform is found.
    """
    extr = metadata.get("extrinsics", {})
    if "T_sensor_in_parent" in extr:
        T = extr["T_sensor_in_parent"]
        if isinstance(T, np.ndarray):
            return T
        else:
            return np.asarray(T, dtype=np.float64)

    raise KeyError(
        "Could not find T_world_from_sensor in metadata (looked for "
        "['extrinsics']['T_sensor_in_parent'] or "
        "['calibration']['extrinsics']['gTl'])."
    )


def transform_sweep(
    sweep: Sweep,
    T: np.ndarray,
    new_frame: str,
    log_applied_transform: bool = True,
) -> Sweep:
    """
    Apply a homogeneous transform to a Sweep and return a new Sweep in transformed coordinates.

    Parameters
    ----------
    sweep : Sweep
        Input sweep in current frame (sweep.frame).
    T : np.ndarray
        (4, 4) homogeneous transform: new_frame <- sweep.frame
    new_frame : str
        Name of the resulting frame, e.g. 'world'.
    log_applied_transform : bool
        If True, store the applied transform into metadata['extrinsics_applied'].

    Returns
    -------
    Sweep
        New sweep with pts transformed into `new_frame`.
    """
    pts = sweep.pts
    xyz = pts[:, :3]  # (N, 3)
    attrs = pts[:, 3:]  # (N, D-3) possibly empty

    N = xyz.shape[0]
    xyz_h = np.hstack([xyz, np.ones((N, 1), dtype=xyz.dtype)])  # (N,4)
    xyz_new = (T @ xyz_h.T).T[:, :3]  # (N,3)

    if attrs.size > 0:
        pts_new = np.hstack([xyz_new, attrs])
    else:
        pts_new = xyz_new

    meta_new = copy.deepcopy(sweep.metadata)
    meta_new["frame"] = new_frame

    if log_applied_transform:
        extr_applied = meta_new.setdefault("extrinsics_applied", {})
        key = f"T_{new_frame}_from_{sweep.frame}"
        extr_applied[key] = T

    return Sweep(pts=pts_new, metadata=meta_new)


def sweep_to_world(sweep: Sweep) -> Sweep:
    """
    Convert a Sweep to 'world' frame based on its metadata.

    If the sweep is already in world frame, it is returned unchanged.
    """
    if sweep.frame == "world":
        return sweep

    T_world_from_sensor = _get_T_world_from_sensor(sweep.metadata)
    return transform_sweep(sweep, T_world_from_sensor, new_frame="world")


def fuse_sweeps_in_world(sweeps: Sequence[Sweep]) -> Sweep:
    """
    Take multiple raw sweeps (possibly from different sensors) and fuse
    them into a single Sweep in a common 'world' frame.

    No temporal alignment or preference is done here: it simply:
      - Transforms each sweep to world frame.
      - Concatenates all points.
      - Aggregates basic metadata (sensors, timestamps).

    Parameters
    ----------
    sweeps : Sequence[Sweep]
        List/tuple of Sweep objects.

    Returns
    -------
    Sweep
        A single Sweep in world frame, containing all fused points.
    """
    if len(sweeps) == 0:
        raise ValueError("fuse_sweeps_in_world: got empty sweep list.")

    world_sweeps: List[Sweep] = [sweep_to_world(s) for s in sweeps]
    pts_list = [s.pts for s in world_sweeps]

    pts_fused = np.vstack(pts_list)

    sensors = [s.metadata.get("sensor") for s in sweeps]
    timestamps = [s.metadata.get("timestamp_ms", 0) for s in sweeps]

    fused_metadata: Dict[str, Any] = {
        "frame": "world",
        "sensors": sensors,
        "timestamps_ms": timestamps,
        "timestamp": timestamps[0] / 1000.0,
        "num_inputs": len(sweeps),
    }

    fused_metadata["sources"] = [
        {
            "sensor": s.metadata.get("sensor"),
            "timestamp_ms": s.metadata.get("timestamp_ms"),
            "timestamp": s.metadata.get("timestamp_ms", 0) / 1000.0,
            "filename": s.metadata.get("filename"),
            "path": s.metadata.get("path"),
            "frame_in": s.frame,
        }
        for s in sweeps
    ]

    return Sweep(pts=pts_fused, metadata=fused_metadata)
