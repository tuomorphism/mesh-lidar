import numpy as np
from lidar_types import Cluster

from tracking.config import (
    TrackingConf,
)


def wrap_angle(a: float) -> float:
    """
    Wrap angle to (-pi, pi].
    """
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def project_to_SO3(R: np.ndarray) -> np.ndarray:
    """
    Project a 3x3 matrix to the nearest proper rotation in SO(3),
    fixing reflection if needed.
    """
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def sanitize_T(T: np.ndarray) -> np.ndarray:
    """
    Sanitize a homogeneous transform by projecting its rotation
    onto SO(3).
    """
    Ts = T.copy()
    Ts[:3, :3] = project_to_SO3(Ts[:3, :3])
    return Ts


def yaw_from_T(T: np.ndarray) -> float:
    """
    Extract yaw (rotation around z) from a 4x4 pose.
    """
    return float(np.arctan2(T[1, 0], T[0, 0]))


def se2_state_from_T(T: np.ndarray, v: float = 0.0, omega: float = 0.0) -> np.ndarray:
    """
    Build 5D EKF state [px, py, psi, v, omega] from pose T and velocities.
    """
    x = float(T[0, 3])
    y = float(T[1, 3])
    psi = yaw_from_T(T)
    return np.array([x, y, psi, v, omega], dtype=float)


def T_from_se2_state(x: np.ndarray, T_template: np.ndarray | None = None) -> np.ndarray:
    """
    Build a 4x4 pose in SE(3) from SE(2) state [px, py, psi].

    Keeps z and roll/pitch from T_template if provided, and overwrites the
    yaw portion of the rotation with psi.
    """
    px, py, psi = float(x[0]), float(x[1]), float(x[2])
    c, s = np.cos(psi), np.sin(psi)

    if T_template is None:
        T = np.eye(4, dtype=float)
    else:
        T = T_template.copy()

    R = T[:3, :3]
    # overwrite yaw part, keep z-axis and roll/pitch structure
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s, c
    T[:3, :3] = project_to_SO3(R)
    T[0, 3] = px
    T[1, 3] = py
    return T


def ctrv_process_model(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Constant-Turn-Rate-and-Velocity (CTRV) model in the xy-plane.

    State:
        x = [px, py, psi, v, omega]
    """
    px, py, psi, v, w = map(float, x)

    if abs(w) < TrackingConf.straight_line_eps:
        # Straight-line motion
        px += v * dt * np.cos(psi)
        py += v * dt * np.sin(psi)
        # psi unchanged except for noise handled by Q in EKF
    else:
        # Circular arc
        psi2 = psi + w * dt
        inv_w = 1.0 / w

        px += v * inv_w * (np.sin(psi2) - np.sin(psi))
        py += v * inv_w * (-np.cos(psi2) + np.cos(psi))
        psi = psi2

    psi = wrap_angle(psi)
    return np.array([px, py, psi, v, w], dtype=float)


def ctrv_jacobian_numeric(x: np.ndarray, dt: float, eps: float = 1e-3) -> np.ndarray:
    """
    Numeric Jacobian F = df/dx at x for the CTRV process model.
    x, f(x) are 5D; F is (5,5).
    """
    f0 = ctrv_process_model(x, dt)
    F = np.zeros((5, 5), dtype=float)
    for i in range(5):
        dx = np.zeros_like(x)
        dx[i] = eps
        fi = ctrv_process_model(x + dx, dt)
        F[:, i] = (fi - f0) / eps
    return F


def ctrv_jacobian_analytic(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Compact analytic Jacobian F = df/dx for CTRV model.

    State:
        x = [px, py, psi, v, omega]
    """
    _, _, psi, v, w = map(float, x)
    F = np.eye(5, dtype=float)

    if abs(w) < TrackingConf.straight_line_eps:
        c = np.cos(psi)
        s = np.sin(psi)

        # px' = px + v dt cos(psi)
        F[0, 2] = -v * dt * s  # d(px')/d(psi)
        F[0, 3] = dt * c  # d(px')/d(v)

        # py' = py + v dt sin(psi)
        F[1, 2] = v * dt * c  # d(py')/d(psi)
        F[1, 3] = dt * s  # d(py')/d(v)

        # psi' = psi  (approximately; true model psi' = psi + w dt,
        #              but for straight-line eps we treat w ≈ 0)
        F[2, 4] = dt  # keep this for smooth transition
        return F

    # General case: nonzero omega
    psi2 = psi + w * dt
    s1, c1 = np.sin(psi), np.cos(psi)
    s2, c2 = np.sin(psi2), np.cos(psi2)

    A = s2 - s1  # term for px
    B = c1 - c2  # term for py

    # dA/dpsi, dB/dpsi
    dA_dpsi = c2 - c1
    dB_dpsi = s2 - s1

    # dA/dw, dB/dw
    dA_dw = c2 * dt
    dB_dw = s2 * dt

    inv_w = 1.0 / w
    inv_w2 = inv_w**2

    # px' = px + v/w * A
    F[0, 2] = v * inv_w * dA_dpsi
    F[0, 3] = inv_w * A
    F[0, 4] = v * (dA_dw * w - A) * inv_w2

    # py' = py + v/w * B
    F[1, 2] = v * inv_w * dB_dpsi
    F[1, 3] = inv_w * B
    F[1, 4] = v * (dB_dw * w - B) * inv_w2

    # psi' = psi + w dt
    F[2, 4] = dt

    return F


def vel_xy_from_state(x: np.ndarray) -> np.ndarray:
    """
    Convert CTRV state x = [px, py, yaw, v, omega] to XY velocity vector.
    """
    yaw = float(x[2])
    v = float(x[3])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([v * c, v * s], dtype=float)


def measurement_noise_R(position_noise: float, direction_noise: float) -> np.ndarray:
    """
    Measurement noise covariance for z = [px, py, yaw].
    """
    return np.diag(
        [
            position_noise**2,
            position_noise**2,
            direction_noise**2,
        ]
    )


def compute_measurement_noise_scaling(cluster: Cluster) -> float:
    """Return multiplicative scaling for measurement noise."""
    N = max(len(cluster.member_indices), 1)

    N_ref = 30.0  # cluster with 30 points gets scale ~1
    scale_pts = np.sqrt(N_ref / min(N, N_ref))

    sx, sy, _ = cluster.geometry.sizes
    size_min = max(sx, sy)
    size_ref = 1.0  # ~1m (pedestrian-size); anything smaller gets scaled up

    scale_size = np.clip(size_ref / max(size_min, 0.2), 1.0, 3.0)

    return max(scale_pts, scale_size)


def extract_pose(cluster: Cluster) -> np.ndarray:
    """
    Extract 4x4 SE3 pose from a cluster's geometry.

    Assumes:
    - cluster.geometry.centroid: (3,)
    - cluster.geometry.rotation: (3,3)
    """
    c = np.asarray(cluster.geometry.centroid, dtype=float).reshape(3)
    R = np.asarray(cluster.geometry.rotation, dtype=float).reshape(3, 3)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = c
    return T


def shape_rel_dev(a: np.ndarray, b: np.ndarray) -> float:
    """
    Average relative deviation between two size vectors (3,).
    """
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    denom = np.maximum(1e-6, 0.5 * (np.abs(a) + np.abs(b)))
    return float(np.mean(np.abs(a - b) / denom))


def meas_from_pose(T: np.ndarray) -> np.ndarray:
    """
    Measurement function h(T): [px, py, yaw].
    """
    return np.array(
        [T[0, 3], T[1, 3], yaw_from_T(T)],
        dtype=float,
    )
