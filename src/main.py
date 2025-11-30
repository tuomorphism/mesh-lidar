from pathlib import Path
from tqdm import tqdm
import pickle
import hashlib
import numpy as np
import trimesh
import open3d as o3d

from visualization.scene_track_visualization import SceneAndTrackViewer
from scene_processing.loader import load_sequence_timesynced
from scene_processing.processor import process_sweeps
from tracking.tracking import Tracker, TrackingConf
from tsdf.classification_track import classify_static_dynamic, StaticDynamicConf
from tsdf.mesh_reconstruction import reconstruct_mesh
from fusion import fuse_sweeps_in_world


def _make_cache_dir(root: Path) -> Path:
    """
    Create (if needed) and return the cache directory for a dataset root.
    """
    cache_dir = root / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _key_to_filename(cache_dir: Path, key: str) -> Path:
    """
    Map an arbitrary string key to a short, stable filename via hashing.
    """
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{h}.pkl"


def load_or_compute(
    root: Path,
    key: str,
    fn,
    args=(),
    kwargs=None,
    force_recompute: bool = False,
):
    """
    Load a value from disk cache if it exists, otherwise compute it.

    Parameters
    ----------
    root : Path
        Dataset root; cache directory is created under this.
    key : str
        Logical key for this computation (include N, config versions, etc.).
    fn : callable
        Function to call on cache miss.
    args : tuple
        Positional args to pass to fn.
    kwargs : dict or None
        Keyword args to pass to fn.
    force_recompute : bool
        If True, ignore cache and recompute.

    Returns
    -------
    Any
        The computed or loaded value.
    """
    cache_dir = _make_cache_dir(root)
    cache_file = _key_to_filename(cache_dir, key)
    kwargs = kwargs or {}

    if cache_file.exists() and not force_recompute:
        print(f"[cache] Loading {key} from {cache_file}")
        with cache_file.open("rb") as f:
            return pickle.load(f)

    print(f"[cache] Computing {key} ...")
    value = fn(*args, **kwargs)
    with cache_file.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[cache] Saved {key} to {cache_file}")

    return value


def main(
    dataset_designation: Path = Path("UrbanIng-V2X/dataset/20241126_0017_crossing1_00"),
    lidar_designations: list[str] = [
        "crossing1_11_lidar",
        "crossing1_12_lidar",
        "crossing1_31_lidar",
        "crossing1_32_lidar",
    ],
    N: int = 200,
    mesh_output_path: Path = Path("./output/mesh.ply"),
    # per-stage recompute flags
    force_recompute_fusion: bool = True,
    force_recompute_processing: bool = False,
    force_recompute_tracking: bool = False,
    force_recompute_mesh: bool = False,
):
    """
    Main pipeline:
      1. Load time-synced multi-LiDAR sweeps
      2. Fuse sweeps into a single world-frame sweep per timestamp
      3. Process sweeps (clustering, features, etc.)
      4. Track objects with EKF-based tracker
      5. Classify static vs dynamic and filter tracks
      6. TSDF mesh reconstruction for static environment
      7. Export mesh and launch viewer
    """
    root = Path("./datasets") / dataset_designation

    # Make sure output dir exists
    mesh_output_path.parent.mkdir(parents=True, exist_ok=True)

    sweep_data = load_sequence_timesynced(root, max_frames=N)
    timestamps = list(sweep_data.keys())
    print(f"Number of frames: {len(timestamps)}")

    def _compute_fused_sweeps():
        combined_sweeps = []
        total_point_count = 0

        for ts in tqdm(timestamps[:N], desc="Fusing sweeps"):
            sd = sweep_data[ts]

            sweeps = [sd[f] for f in lidar_designations]

            combined_sweep = fuse_sweeps_in_world(sweeps)
            combined_sweeps.append(combined_sweep)
            total_point_count += combined_sweep.pts.shape[0]

        print(f"Total point count: {total_point_count} points")
        return combined_sweeps

    combined_sweeps = load_or_compute(
        root=root,
        key=f"fused_sweeps_N={N}",
        fn=_compute_fused_sweeps,
        args=(),
        kwargs=None,
        force_recompute=force_recompute_fusion,
    )

    processed = load_or_compute(
        root=root,
        key=f"processed_sweeps_N={N}",
        fn=process_sweeps,
        args=(combined_sweeps[:N],),
        kwargs=None,
        force_recompute=force_recompute_processing,
    )

    def _compute_tracking():
        print("Creating tracks")
        tracker_conf = TrackingConf()
        tracker = Tracker(tracker_conf)
        result = tracker.fit(processed)

        classification_conf = StaticDynamicConf()
        for history in tqdm(result.histories, desc="Classifying static/dynamic"):
            classify_static_dynamic(history, classification_conf, processed)

        return result

    tracking_result = load_or_compute(
        root=root,
        key=f"tracking_result_N={N}",
        fn=_compute_tracking,
        args=(),
        kwargs=None,
        force_recompute=force_recompute_tracking,
    )

    def _compute_mesh():
        print("Reconstructing mesh")
        mesh_result = reconstruct_mesh(processed, tracking_result.histories)
        verts, faces, normals = mesh_result.extract_mesh()
        return verts, faces, normals

    verts, faces, normals = load_or_compute(
        root=root,
        key=f"mesh_N={N}",
        fn=_compute_mesh,
        args=(),
        kwargs=None,
        force_recompute=force_recompute_mesh,
    )

    mesh_data = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh_data.export(mesh_output_path)
    print(f"Mesh exported to: {mesh_output_path}")

    mesh = o3d.io.read_triangle_mesh(f"{mesh_output_path}")

    viewer = SceneAndTrackViewer(
        processed,
        tracking_result=tracking_result,
        min_diag=0.1,
        static_mesh=mesh,
        static_mesh_color=np.array([0.7, 0.7, 0.7]),
    )
    viewer.run()


if __name__ == "__main__":
    main()
