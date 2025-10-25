
from loader import ProcessedScene

def accumulate_points(base_scene: ProcessedScene, source_scenes: ProcessedScene):
    # Create cubic voxels at base_scene points
    base_scene.