import numpy as np
from scipy.spatial import KDTree
from scene_processing.config import Config


def nn_flow(X, Y, dt):
    """
    Very simple nearest-neighbour scene flow.

    X: (N, 3) points at time t  (in SAME frame as Y)
    Y: (M, 3) points at time t+dt
    dt: time delta (seconds)
    max_dist: maximum distance for valid matches

    Returns:
      vel: (N, 3) velocity estimates, NaN where no valid nearest neighbour match
      valid: (N,) boolean mask of points with a valid nearest neighbour match
    """

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)

    # Build the KD-tree and query the top 1 closest point
    tree = KDTree(Y)
    dists, idxs = tree.query(X, k=1)

    # Max dist
    valid = dists < Config.velocity_max_dist
    disp = np.full_like(X, np.nan)
    vel = np.full_like(X, np.nan)

    # Displacement and velocities, simple finite difference velocity estimation v = d / t.
    disp[valid] = Y[idxs[valid]] - X[valid]
    vel[valid] = disp[valid] / max(dt, 1e-6)

    vel[~valid] = 0.0

    return vel, valid
