import numpy as np

_EPS = 1e-8


def _skew(w):
    wx, wy, wz = w
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=float)


def lin_speed_from_twist(twist_hat: np.ndarray) -> float:
    vx, vy, vz = twist_hat[0, 3], twist_hat[1, 3], twist_hat[2, 3]
    return float(np.sqrt(vx * vx + vy * vy + vz * vz))


def predict_se2_CTRV(T_w: np.ndarray, v: float, psi: float, psidot: float, dt: float):
    # Depending on the anlular velocity, we either make approximation of straight line movement
    # or an approximation of movement along an arc
    # This is known as Constant Turn Rate and Velocity (CTRV) approximation
    if abs(psidot) < 1e-4:
        px = T_w[0, 3] + v * dt * np.cos(psi)
        py = T_w[1, 3] + v * dt * np.sin(psi)
        psi_new = psi
    else:
        R = v / psidot
        dpsi = psidot * dt
        px = T_w[0, 3] + R * (np.sin(psi + dpsi) - np.sin(psi))
        py = T_w[1, 3] - R * (np.cos(psi + dpsi) - np.cos(psi))
        psi_new = psi + dpsi

    # write back into a 4x4 SE(2) pose
    Tp = T_w.copy()
    c, s = np.cos(psi_new), np.sin(psi_new)
    Tp[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Tp[0, 3], Tp[1, 3] = px, py
    return Tp


def project_to_SO3(R):
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def sanitize_T(T):
    Ts = T.copy()
    Ts[:3, :3] = project_to_SO3(Ts[:3, :3])
    return Ts


def rigid_inverse(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def log_SO3(R):
    R = project_to_SO3(R)
    tr = np.clip(np.trace(R), -1.0, 3.0)
    cos_th = 0.5 * (tr - 1.0)
    cos_th = np.clip(cos_th, -1.0, 1.0)
    theta = np.arccos(cos_th)
    if theta < _EPS:
        w_hat = 0.5 * (R - R.T)
        return np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], float)
    w_hat = 0.5 * (R - R.T)
    w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], float)
    return w * (theta / (np.linalg.norm(w) + 1e-12))


def exp_SO3(omega):
    theta = np.linalg.norm(omega)
    I = np.eye(3)
    W = _skew(omega)
    if theta < _EPS:
        return I + W + 0.5 * (W @ W)
    th2 = theta * theta
    s_over_th = np.sin(theta) / theta
    one_minus_c_over_th2 = (1.0 - np.cos(theta)) / th2
    return I + s_over_th * W + one_minus_c_over_th2 * (W @ W)


def _Jl_SO3(omega):
    theta = np.linalg.norm(omega)
    I = np.eye(3)
    W = _skew(omega)
    if theta < _EPS:
        return I + 0.5 * W + (1.0 / 6.0) * (W @ W)
    th2 = theta * theta
    A = (1.0 - np.cos(theta)) / th2
    B = (theta - np.sin(theta)) / (theta * th2)
    return I + A * W + B * (W @ W)


def _Jl_SO3_inv(omega):
    theta = np.linalg.norm(omega)
    I = np.eye(3)
    W = _skew(omega)
    if theta < _EPS:
        return I - 0.5 * W + (1.0 / 12.0) * (W @ W)
    half = 0.5 * theta
    cot_half = np.cos(half) / np.sin(half)
    return I - 0.5 * W + ((1.0 - 0.5 * theta * cot_half) / (theta**2)) * (W @ W)


def hat_se3(omega, rho):
    Xi = np.zeros((4, 4))
    Xi[:3, :3] = _skew(omega)
    Xi[:3, 3] = rho
    return Xi


def exp_SE3(Xi):
    omega = np.array([Xi[2, 1], Xi[0, 2], Xi[1, 0]], float)
    rho = Xi[:3, 3].astype(float)
    R = exp_SO3(omega)
    V = _Jl_SO3(omega)
    t = V @ rho
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def log_SE3(T):
    R = T[:3, :3]
    t = T[:3, 3].astype(float)
    omega = log_SO3(R)
    Vinv = _Jl_SO3_inv(omega)
    rho = Vinv @ t
    return omega, rho


def pose_distance_SE3(T1, T2, lambda_m_per_rad=2.0, Wt=None, Wr=None):
    Trel = rigid_inverse(T1) @ (T2)
    omega, v = log_SE3(Trel)
    if Wt is None:
        Wt = np.eye(3)
    if Wr is None:
        Wr = (lambda_m_per_rad**2) * np.eye(3)
    return float(np.sqrt(v @ (Wt @ v) + omega @ (Wr @ omega)))
