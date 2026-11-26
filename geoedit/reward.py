"""Per-pixel likelihood-ratio map (P3 §5): "does this pixel look more like the target?"

For an image ``x`` and independent noises ``eps_i``::

    z_i        = a x + b eps_i
    l_p        = 1/2 * sum_i ( ||eps_hat_src(z_i) - eps_i||_p^2
                             - ||eps_hat_tar(z_i) - eps_i||_p^2 )

``||.||_p^2`` is the sum over the 4 latent channels of pixel ``p``.  A pixel on the target
manifold is better predicted by the target branch, so ``l_p > 0`` means "more target-like".
``l = 0`` when the two prompts are identical (T-R1).  Cost: ``2 * len(eps_list)`` forwards.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .model import SDTurbo


def lr_map(
    model: SDTurbo,
    x: torch.Tensor,
    t: float,
    eps_list: Sequence[torch.Tensor],
    emb_src: torch.Tensor,
    emb_tar: torch.Tensor,
) -> torch.Tensor:
    """``[64, 64]`` likelihood-ratio map, averaged over the given noises."""
    if len(eps_list) == 0:
        raise ValueError("eps_list is empty")
    a, b = model.ab(t)
    total = None
    for eps in eps_list:
        z = a * x + b * eps
        src = model.x0_pred(z, t, emb_src)          # 1 forward
        tar = model.x0_pred(z, t, emb_tar)          # 1 forward
        src_eps = (z - a * src) / b
        tar_eps = (z - a * tar) / b
        diff = ((src_eps - eps) ** 2).sum(dim=1) - ((tar_eps - eps) ** 2).sum(dim=1)
        total = diff if total is None else total + diff
    assert total is not None
    return 0.5 * total / float(len(eps_list))


def refine(w: torch.Tensor, delta_l: torch.Tensor, pool: int = 3) -> torch.Tensor:
    """``w_1 = w_0 * 1[ smoothed(delta_l) > 0 ]``: drop pixels that did not get more target-like."""
    import torch.nn.functional as F

    if delta_l.dim() == 4 and delta_l.shape[0] == 1:
        delta_l = delta_l[0]
    if delta_l.dim() == 3 and delta_l.shape[0] == 1:
        delta_l = delta_l[0]
    if pool > 1:
        smoothed = F.avg_pool2d(delta_l[None, None], kernel_size=pool, stride=1,
                                padding=pool // 2, count_include_pad=False)[0, 0]
    else:
        smoothed = delta_l
    return w * (smoothed > 0).float()
