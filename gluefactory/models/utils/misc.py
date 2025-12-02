import math
from typing import List, Optional, Tuple, Union

import torch


def to_sequence(map):
    return map.flatten(-2).transpose(-1, -2)


def to_map(sequence):
    n = sequence.shape[-2]
    e = math.isqrt(n)
    assert e * e == n
    assert e * e == n
    sequence.transpose(-1, -2).unflatten(-1, [e, e])


def pad_to_length(
    x,
    length: int,
    pad_dim: int = -2,
    mode: Union[str, bool] = "zeros",  # zeros, ones, random, random_c, minus_one, False
    bounds: Tuple[Optional[int], Optional[int]] = (None, None),
):
    shape = list(x.shape)
    d = x.shape[pad_dim]
    assert d <= length
    if d == length:
        return x
    shape[pad_dim] = length - d

    low, high = bounds
    mode_key = mode.lower() if isinstance(mode, str) else mode

    if mode_key == "zeros":
        xn = torch.zeros(*shape, device=x.device, dtype=x.dtype)
    elif mode_key == "ones":
        xn = torch.ones(*shape, device=x.device, dtype=x.dtype)
    elif mode_key == "minus_one":
        xn = torch.full(shape, -1, device=x.device, dtype=x.dtype)
    elif mode_key is False or mode_key == "false":
        xn = torch.zeros(*shape, device=x.device, dtype=x.dtype)
    elif mode_key == "random":
        low = low if low is not None else x.min()
        high = high if high is not None else x.max()
        xn = torch.empty(*shape, device=x.device).uniform_(low, high)
    elif mode_key == "random_c":
        low, high = bounds  # we use the bounds as fallback for empty seq.
        xn = torch.cat(
            [
                torch.empty(*shape[:-1], 1, device=x.device).uniform_(
                    x[..., i].min() if d > 0 else low,
                    x[..., i].max() if d > 0 else high,
                )
                for i in range(shape[-1])
            ],
            dim=-1,
        )
    else:
        raise ValueError(mode)
    return torch.cat([x, xn], dim=pad_dim)


def pad_and_stack(
    sequences: List[torch.Tensor],
    length: Optional[int] = None,
    pad_dim: int = -2,
    **kwargs,
):
    if length is None:
        length = max([x.shape[pad_dim] for x in sequences])

    y = torch.stack([pad_to_length(x, length, pad_dim, **kwargs) for x in sequences], 0)
    return y
