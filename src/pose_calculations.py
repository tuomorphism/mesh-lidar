import numpy as np


def rigid_inverse(T: np.ndarray) -> np.ndarray:
    R = project_to_SO3(T[:3, :3])
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def sanitize_T(T: np.ndarray) -> np.ndarray:
    Ts = T.copy()
    Ts[:3, :3] = project_to_SO3(Ts[:3, :3])
    return Ts


def _V_inv(omega):
    return _left_Jacobian_SO3_inv(omega)


def pose_distance_SE3(T1, T2, lambda_m_per_rad=2.0, Wt=None, Wr=None):
    """
    True SE(3) metric from the Lie log, with weights:
      d^2 = v^T Wt v + omega^T Wr omega
    Defaults: Wt = I, Wr = (lambda^2) I   (lambda converts radians -> meters)
    """
    # relative transform
    R1 = T1[:3, :3]
    t1 = T1[:3, 3]
    R2 = T2[:3, :3]
    t2 = T2[:3, 3]
    Trel = np.eye(4)
    Trel[:3, :3] = R1.T @ R2
    Trel[:3, 3] = R1.T @ (t2 - t1)

    omega, v = log_SE3(Trel)

    if Wt is None:
        Wt = np.eye(3)
    if Wr is None:
        Wr = (lambda_m_per_rad**2) * np.eye(3)

    return float(np.sqrt(v @ (Wt @ v) + omega @ (Wr @ omega)))


def _rot_exp(omega):
    """
    Rodrigues' formula with small-angle fallback.
    omega: R^3 (angle-axis vector, length = theta)
    """
    theta = np.linalg.norm(omega)
    I = np.eye(3)
    if theta < 1e-8:
        # R ≈ I + [w]
        W = _skew(omega)
        return I + W
    W = _skew(omega)
    th2 = theta * theta
    s_over_th = np.sin(theta) / theta
    one_minus_c_over_th2 = (1.0 - np.cos(theta)) / th2
    return I + s_over_th * W + one_minus_c_over_th2 * (W @ W)


def _V(omega):
    return _left_Jacobian_SO3(omega)


def _skew(w):
    wx, wy, wz = w
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=float)


def project_to_SO3(R):
    # Polar/SVD projection to nearest proper rotation
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def log_SO3(R):
    R = project_to_SO3(R)
    tr = np.clip(np.trace(R), -1.0, 3.0)
    cos_th = 0.5 * (tr - 1.0)
    cos_th = np.clip(cos_th, -1.0, 1.0)
    theta = np.arccos(cos_th)
    if theta < 1e-8:
        return np.zeros(3)
    w_hat = (R - R.T) * 0.5
    omega = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
    omega = omega * (theta / (np.linalg.norm(omega) + 1e-12))
    return omega


def exp_SO3(omega):
    theta = np.linalg.norm(omega)
    I = np.eye(3)
    if theta < 1e-8:
        W = _skew(omega)
        return I + W + 0.5 * (W @ W)
    W = _skew(omega)
    th2 = theta * theta
    s_over_th = np.sin(theta) / theta
    one_minus_c_over_th2 = (1.0 - np.cos(theta)) / th2
    return I + s_over_th * W + one_minus_c_over_th2 * (W @ W)


def _left_Jacobian_SO3(omega):
    theta = np.linalg.norm(omega)
    I = np.eye(3)
    if theta < 1e-8:
        W = _skew(omega)
        return I + 0.5 * W + (1.0 / 6.0) * (W @ W)
    W = _skew(omega)
    th2 = theta * theta
    A = (1.0 - np.cos(theta)) / th2
    B = (theta - np.sin(theta)) / (theta * th2)
    return I + A * W + B * (W @ W)


def _left_Jacobian_SO3_inv(omega):
    theta = np.linalg.norm(omega)
    I = np.eye(3)
    W = _skew(omega)
    if theta < 1e-8:
        return I - 0.5 * W + (1.0 / 12.0) * (W @ W)
    half = 0.5 * theta
    cot_half = np.cos(half) / np.sin(half)
    Wn = _skew(omega / theta)
    return (
        I
        - 0.5 * (Wn * theta)
        + (1 - 0.5 * theta * cot_half) / (theta**2) * ((Wn * theta) @ (Wn * theta))
    )


def hat_se3(omega, rho):
    Xi = np.zeros((4, 4))
    Xi[:3, :3] = _skew(omega)
    Xi[:3, 3] = rho
    return Xi


def exp_SE3(Xi):
    omega = np.array([Xi[2, 1], Xi[0, 2], Xi[1, 0]], float)
    rho = Xi[:3, 3].astype(float)
    R = exp_SO3(omega)
    V = _left_Jacobian_SO3(omega)
    t = V @ rho
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def log_SE3(T):
    R = project_to_SO3(T[:3, :3])
    t = T[:3, 3].astype(float)
    omega = log_SO3(R)
    Vinv = _left_Jacobian_SO3_inv(omega)
    rho = Vinv @ t
    return omega, rho
