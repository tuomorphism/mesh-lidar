import numpy as np


def _skew(w):
    wx, wy, wz = w
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=float)


def _rot_log(R):
    # returns angle-axis vector omega (length = theta)
    tr = np.clip(np.trace(R), -1.0, 3.0)
    cos_th = (tr - 1.0) * 0.5
    cos_th = np.clip(cos_th, -1.0, 1.0)
    theta = np.arccos(cos_th)
    if theta < 1e-8:
        # very small rotation -> use first order
        return np.zeros(3, float)
    # robust axis from skew(R)
    w_hat = (R - R.T) * 0.5
    # axis components (note: for small angles this is safe because we guarded above)
    wx = w_hat[2, 1]
    wy = w_hat[0, 2]
    wz = w_hat[1, 0]
    # Scale to get angle-axis length = theta
    # sin(theta) may be tiny; clamp
    s = np.sin(theta)
    if abs(s) < 1e-8:
        return np.zeros(3, float)
    axis = (1.0 / (2.0 * s)) * np.array([wx, wy, wz], float)
    return theta * axis


def _V_inv(omega):
    theta = np.linalg.norm(omega)
    I = np.eye(3)
    if theta < 1e-6:
        # series: I - 1/2 [w] + 1/12 [w]^2
        W = _skew(omega)
        return I - 0.5 * W + (1.0 / 12.0) * (W @ W)
    W = _skew(omega)
    th2 = theta * theta
    a = 0.5
    b = (1.0 / theta**2) * (1 - (theta * np.sin(theta)) / (2 * (1 - np.cos(theta))))
    # A common stable closed form for V^{-1}:
    # V^{-1} = I - 1/2 [w] + b [w]^2
    return I - a * W + b * (W @ W)


def log_SE3(T):
    """
    Returns (omega, v) where:
      omega: angle-axis vector (R^3), length = rotation angle in rad
      v    : 'translation' in the Lie algebra (R^3), such that exp([omega]_x, v) = T.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    omega = _rot_log(R)
    Vinv = _V_inv(omega)
    v = Vinv @ t
    return omega, v


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
    """
    Left-Jacobian V(omega) for SE(3) exponential.
    For small theta: V ≈ I + 1/2 [w] + 1/6 [w]^2
    """
    theta = np.linalg.norm(omega)
    I = np.eye(3)
    W = _skew(omega)
    if theta < 1e-6:
        return I + 0.5 * W + (1.0 / 6.0) * (W @ W)
    th2 = theta * theta
    A = (1.0 - np.cos(theta)) / th2
    B = (theta - np.sin(theta)) / (theta * th2)
    return I + A * W + B * (W @ W)


def hat_se3(omega, v):
    """
    Build 4x4 twist matrix from (omega, v).
    """
    Xi = np.zeros((4, 4), dtype=float)
    Xi[:3, :3] = _skew(omega)
    Xi[:3, 3] = v
    return Xi


def exp_SE3(arg, v=None):
    """
    Exponential map to SE(3).

    Usage:
      - exp_SE3(Xi_hat): Xi_hat is 4x4 in se(3)
      - exp_SE3(omega, v): omega,v are 3-vectors

    Returns:
      4x4 homogeneous transform T = exp(Xi_hat)
    """
    if v is None:
        # arg is a 4x4 twist (hat) matrix
        Xi_hat = np.asarray(arg, dtype=float)
        W = Xi_hat[:3, :3]
        omega = np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)
        v = Xi_hat[:3, 3]
    else:
        omega = np.asarray(arg, dtype=float)
        v = np.asarray(v, dtype=float)

    R = _rot_exp(omega)
    V = _V(omega)
    t = V @ v

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
