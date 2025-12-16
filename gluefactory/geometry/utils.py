from typing import Tuple

import numpy as np
import torch


def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points, eps=0.0):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
        eps: Epsilon value to prevent zero division.
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)


def batched_eye_like(x: torch.Tensor, n: int):
    """Create a batch of identity matrices.
    Args:
        x: a reference torch.Tensor whose batch dimension will be copied.
        n: the size of each identity matrix.
    Returns:
        A torch.Tensor of size (B, n, n), with same dtype and device as x.
    """
    return torch.eye(n).to(x)[None].repeat(len(x), 1, 1)


def skew_symmetric(v):
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3)."""
    z = torch.zeros_like(v[..., 0])
    M = torch.stack(
        [
            z,
            -v[..., 2],
            v[..., 1],
            v[..., 2],
            z,
            -v[..., 0],
            -v[..., 1],
            v[..., 0],
            z,
        ],
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))
    return M


def transform_points(T, points):
    return from_homogeneous(to_homogeneous(points) @ T.transpose(-1, -2))


def is_inside(pts, shape):
    return (pts > 0).all(-1) & (pts < shape[:, None]).all(-1)


def so3exp_map(w, eps: float = 1e-7):
    """Compute rotation matrices from batched twists.
    Args:
        w: batched 3D axis-angle vectors of size (..., 3).
    Returns:
        A batch of rotation matrices of size (..., 3, 3).
    """
    theta = w.norm(p=2, dim=-1, keepdim=True)
    small = theta < eps
    div = torch.where(small, torch.ones_like(theta), theta)
    W = skew_symmetric(w / div)
    theta = theta[..., None]  # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(small[..., None], W, res)  # first-order Taylor approx
    return torch.eye(3).to(W) + res


@torch.jit.script
def distort_points(pts, dist):
    """Distort normalized 2D coordinates
    and check for validity of the distortion model.
    """
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]
    undist = pts
    valid = torch.ones(pts.shape[:-1], device=pts.device, dtype=torch.bool)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts**2, -1, keepdim=True)
        radial = k1 * r2 + k2 * r2**2
        undist = undist + pts * radial

        # The distortion model is supposedly only valid within the image
        # boundaries. Because of the negative radial distortion, points that
        # are far outside of the boundaries might actually be mapped back
        # within the image. To account for this, we discard points that are
        # beyond the inflection point of the distortion model,
        # e.g. such that d(r + k_1 r^3 + k2 r^5)/dr = 0
        limited = ((k2 > 0) & ((9 * k1**2 - 20 * k2) > 0)) | ((k2 <= 0) & (k1 > 0))
        limit = torch.abs(
            torch.where(
                k2 > 0,
                (torch.sqrt(9 * k1**2 - 20 * k2) - 3 * k1) / (10 * k2),
                1 / (3 * k1),
            )
        )
        valid = valid & torch.squeeze(~limited | (r2 < limit), -1)

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            uv = torch.prod(pts, -1, keepdim=True)
            undist = undist + 2 * p12 * uv + p21 * (r2 + 2 * pts**2)
            # TODO: handle tangential boundaries

    return undist, valid


def distort_points_fisheye_kb4(
    pts: torch.Tensor, dist: torch.Tensor, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Distort normalized 2D coordinates using the OpenCV/Colmap fisheye KB4 model.

    Args:
        pts: normalized coordinates with shape (..., 2).
        dist: distortion coefficients with shape (..., 4) as (k1, k2, k3, k4).
        eps: threshold to handle r=0 safely.
    Returns:
        pts_d: distorted normalized coordinates with shape (..., 2).
        valid: boolean mask with shape (...) indicating finite output.
    """
    assert pts.shape[-1] == 2
    dist = dist.unsqueeze(-2)  # add point dimension for broadcasting
    ndist = dist.shape[-1]

    if ndist == 0:
        valid = torch.ones(pts.shape[:-1], device=pts.device, dtype=torch.bool)
        return pts, valid

    k = dist[..., :4]
    if k.shape[-1] < 4:
        k = torch.cat(
            [
                k,
                torch.zeros(
                    k.shape[:-1] + (4 - k.shape[-1],), device=k.device, dtype=k.dtype
                ),
            ],
            dim=-1,
        )
    k1, k2, k3, k4 = k.split(1, dim=-1)

    r = torch.norm(pts, p=2, dim=-1, keepdim=True)
    theta = torch.atan(r)
    theta2 = theta * theta
    theta3 = theta * theta2
    theta5 = theta3 * theta2
    theta7 = theta5 * theta2
    theta9 = theta7 * theta2
    theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9

    scale = torch.ones_like(r)
    valid_r = r > eps
    scale = torch.where(valid_r, theta_d / r, scale)
    pts_d = pts * scale
    valid = torch.isfinite(pts_d).all(dim=-1)
    return pts_d, valid


def undistort_points_fisheye_kb4(
    pts_d: torch.Tensor,
    dist: torch.Tensor,
    max_iters: int = 10,
    tol: float = 1e-12,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert the OpenCV/Colmap fisheye KB4 distortion model (Newton iterations on theta).

    Args:
        pts_d: distorted normalized coordinates with shape (..., 2).
        dist: distortion coefficients with shape (..., 4) as (k1, k2, k3, k4).
    Returns:
        pts: undistorted normalized coordinates with shape (..., 2).
        valid: boolean mask with shape (...) indicating finite output.
    """
    assert pts_d.shape[-1] == 2
    dist = dist.unsqueeze(-2)  # add point dimension for broadcasting
    ndist = dist.shape[-1]

    if ndist == 0:
        valid = torch.ones(pts_d.shape[:-1], device=pts_d.device, dtype=torch.bool)
        return pts_d, valid

    k = dist[..., :4]
    if k.shape[-1] < 4:
        k = torch.cat(
            [
                k,
                torch.zeros(
                    k.shape[:-1] + (4 - k.shape[-1],),
                    device=k.device,
                    dtype=k.dtype,
                ),
            ],
            dim=-1,
        )
    k1, k2, k3, k4 = k.split(1, dim=-1)

    theta_d = torch.norm(pts_d, p=2, dim=-1, keepdim=True)
    theta = theta_d.clone()
    active = theta_d > eps

    for _ in range(max_iters):
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        f = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8) - theta_d
        fp = 1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta4 + 7.0 * k3 * theta6 + 9.0 * k4 * theta8
        step = f / fp

        theta_new = theta - step
        theta = torch.where(active, theta_new, theta)
        active = active & (torch.abs(step) >= tol)
        if not bool(active.any()):
            break

    r = torch.tan(theta)
    scale = torch.ones_like(theta_d)
    valid_r = theta_d > eps
    scale = torch.where(valid_r, r / theta_d, scale)
    pts = pts_d * scale
    valid = torch.isfinite(pts).all(dim=-1)
    return pts, valid


@torch.jit.script
def J_distort_points(pts, dist):
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]

    J_diag = torch.ones_like(pts)
    J_cross = torch.zeros_like(pts)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts**2, -1, keepdim=True)
        uv = torch.prod(pts, -1, keepdim=True)
        radial = k1 * r2 + k2 * r2**2
        d_radial = 2 * k1 + 4 * k2 * r2
        J_diag += radial + (pts**2) * d_radial
        J_cross += uv * d_radial

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            J_diag += 2 * p12 * pts.flip(-1) + 6 * p21 * pts
            J_cross += 2 * p12 * pts + 2 * p21 * pts.flip(-1)

    J = torch.diag_embed(J_diag) + torch.diag_embed(J_cross).flip(-1)
    return J


def get_image_coords(img):
    h, w = img.shape[-2:]
    return (
        torch.stack(
            torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=img.device),
                torch.arange(w, dtype=torch.float32, device=img.device),
                indexing="ij",
            )[::-1],
            dim=0,
        ).permute(1, 2, 0)
    )[None] + 0.5
