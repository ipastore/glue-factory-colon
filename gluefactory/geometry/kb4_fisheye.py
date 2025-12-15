"""
Reference implementation of the OpenCV/Colmap fisheye KB4 camera model.

Distortion model (OpenCV fisheye):
  - x,y are normalized camera coordinates (x=X/Z, y=Y/Z)
  - r = sqrt(x^2 + y^2)
  - theta = atan(r)
  - theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
  - x_d = x * (theta_d / r), y_d = y * (theta_d / r)    (with safe handling for r=0)

Projection:
  u = fx * x_d + cx
  v = fy * y_d + cy
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _parse_K(K: np.ndarray | Iterable[float]) -> Tuple[float, float, float, float]:
    K = np.asarray(K)
    if K.shape == (3, 3):
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        return fx, fy, cx, cy
    K = K.reshape(-1)
    if K.shape[0] != 4:
        raise ValueError("K must be a 3x3 matrix or a 4-vector (fx, fy, cx, cy).")
    fx, fy, cx, cy = (float(x) for x in K.tolist())
    return fx, fy, cx, cy


def _poly_theta_d(theta: np.ndarray, D: np.ndarray | Iterable[float]) -> np.ndarray:
    k1, k2, k3, k4 = (float(x) for x in np.asarray(D).reshape(-1)[:4].tolist())
    theta2 = theta * theta
    theta3 = theta * theta2
    theta5 = theta3 * theta2
    theta7 = theta5 * theta2
    theta9 = theta7 * theta2
    return theta + k1 * theta3 + k2 * theta5 +  k3 * theta7 + k4 * theta9


def distort_kb4(xy: np.ndarray, D: np.ndarray | Iterable[float], eps: float = 1e-12) -> np.ndarray:
    """
    Distort normalized coordinates using the OpenCV fisheye KB4 model.

    Args:
        xy: (..., 2) normalized points (x=X/Z, y=Y/Z)
        D: (4,) distortion parameters (k1, k2, k3, k4)
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[-1] != 2:
        raise ValueError("xy must have shape (..., 2).")

    r = np.linalg.norm(xy, axis=-1)
    theta = np.arctan(r)
    theta_d = _poly_theta_d(theta, D)

    scale = np.ones_like(r)
    valid = r > eps
    scale[valid] = theta_d[valid] / r[valid]
    return xy * scale[..., None]


def undistort_kb4(
    xy_d: np.ndarray,
    D: np.ndarray | Iterable[float],
    *,
    max_iters: int = 10,
    tol: float = 1e-12,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Invert fisheye KB4 distortion (iterative Newton solve on theta).

    Args:
        xy_d: (..., 2) distorted normalized points.
        D: (4,) distortion parameters (k1, k2, k3, k4).
    """
    xy_d = np.asarray(xy_d, dtype=np.float64)
    if xy_d.shape[-1] != 2:
        raise ValueError("xy_d must have shape (..., 2).")

    k1, k2, k3, k4 = (float(x) for x in np.asarray(D).reshape(-1)[:4].tolist())
    theta_d = np.linalg.norm(xy_d, axis=-1)
    theta = theta_d.copy()

    active = theta_d > eps
    if np.any(active):
        for _ in range(max_iters):
            th = theta[active]
            th2 = th * th
            th4 = th2 * th2
            th6 = th4 * th2
            th8 = th4 * th4

            f = th * (1.0 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8) - theta_d[active]
            fp = 1.0 + 3.0 * k1 * th2 + 5.0 * k2 * th4 + 7.0 * k3 * th6 + 9.0 * k4 * th8

            step = f / fp
            theta[active] = th - step
            active[active] = np.abs(step) >= tol
            if not np.any(active):
                break

    r = np.tan(theta)
    scale = np.ones_like(theta_d)
    valid = theta_d > eps
    scale[valid] = r[valid] / theta_d[valid]
    return xy_d * scale[..., None]


def project_kb4(XYZ: np.ndarray, K: np.ndarray | Iterable[float], D: np.ndarray | Iterable[float]) -> np.ndarray:
    """
    Project 3D points in the camera frame (Z>0) to pixel coordinates.

    Args:
        XYZ: (..., 3) 3D points in camera coordinates.
        K: intrinsics (3x3) or (fx, fy, cx, cy).
        D: (4,) distortion parameters (k1, k2, k3, k4).
    """
    XYZ = np.asarray(XYZ, dtype=np.float64)
    if XYZ.shape[-1] != 3:
        raise ValueError("XYZ must have shape (..., 3).")
    fx, fy, cx, cy = _parse_K(K)

    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    x = X / Z
    y = Y / Z
    xy_d = distort_kb4(np.stack([x, y], axis=-1), D)
    u = fx * xy_d[..., 0] + cx
    v = fy * xy_d[..., 1] + cy
    return np.stack([u, v], axis=-1)


def unproject_kb4(uv: np.ndarray, K: np.ndarray | Iterable[float], D: np.ndarray | Iterable[float]) -> np.ndarray:
    """
    Unproject pixels to homogeneous rays with z=1 in the camera frame.

    Args:
        uv: (..., 2) pixel coordinates.
        K: intrinsics (3x3) or (fx, fy, cx, cy).
        D: (4,) distortion parameters (k1, k2, k3, k4).
    """
    uv = np.asarray(uv, dtype=np.float64)
    if uv.shape[-1] != 2:
        raise ValueError("uv must have shape (..., 2).")
    fx, fy, cx, cy = _parse_K(K)

    x_d = (uv[..., 0] - cx) / fx
    y_d = (uv[..., 1] - cy) / fy
    xy = undistort_kb4(np.stack([x_d, y_d], axis=-1), D)
    return np.concatenate([xy, np.ones_like(xy[..., :1])], axis=-1)


if __name__ == "__main__":
    # Tiny self-check: distort/undistort are approximate inverses, and unproject(project(XYZ)) recovers X/Z, Y/Z.
    rng = np.random.default_rng(0)
    K = np.array([[500.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.02, 0.003, -0.0005, 0.00008], dtype=np.float64)

    xy = rng.normal(size=(1000, 2)).astype(np.float64) * 0.4
    xy_d = distort_kb4(xy, D)
    xy_u = undistort_kb4(xy_d, D)
    err_xy = np.max(np.linalg.norm(xy_u - xy, axis=-1))
    assert err_xy < 5e-9, f"distort/undistort mismatch: max={err_xy}"

    XYZ = rng.normal(size=(1000, 3)).astype(np.float64)
    XYZ[:, 2] = np.abs(XYZ[:, 2]) + 0.5
    uv = project_kb4(XYZ, K, D)
    rays = unproject_kb4(uv, K, D)
    xy_true = XYZ[:, :2] / XYZ[:, 2:3]
    err_ray = np.max(np.linalg.norm(rays[:, :2] - xy_true, axis=-1))
    assert err_ray < 5e-9, f"project/unproject mismatch: max={err_ray}"

    print("kb4_fisheye.py self-check: OK")
