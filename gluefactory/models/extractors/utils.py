import torch


def filter_keypoints_by_specular_mask(
    keypoints: torch.Tensor,
    specular_mask: torch.Tensor | None,
    *values: torch.Tensor | None,
    image_size=None,
    keypoint_offset: float = 0.5,
):
    if specular_mask is None or keypoints.numel() == 0:
        return (keypoints, *values)

    mask = specular_mask
    if image_size is not None:
        w, h = image_size
        mask = mask[..., : int(h), : int(w)]
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    mask = mask.to(device=keypoints.device, dtype=torch.bool)

    h, w = mask.shape[-2:]
    xy = keypoints - keypoint_offset
    x0 = torch.floor(xy[:, 0]).long()
    x1 = torch.ceil(xy[:, 0]).long()
    y0 = torch.floor(xy[:, 1]).long()
    y1 = torch.ceil(xy[:, 1]).long()
    inside = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    keep = torch.zeros_like(inside)
    if inside.any():
        keep_inside = (
            mask[y0[inside], x0[inside]]
            & mask[y0[inside], x1[inside]]
            & mask[y1[inside], x0[inside]]
            & mask[y1[inside], x1[inside]]
        )
        keep[inside] = keep_inside

    filtered = [keypoints[keep]]
    for value in values:
        filtered.append(None if value is None else value[keep])
    return tuple(filtered)
