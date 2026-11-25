"""Noise-side tricks (P3 §4): source-consistent (fixed-point) noise, antithetic noise, draw
selection.

``fixed_point_eps`` chases the noise for which the source branch reconstructs ``x_s`` exactly.
At its fixed point ``e_src = 0``, and since ``A = R + e_src`` (§2.1 of the idea doc) the SDS
and DDS directions coincide; the background residual is then only "what the prompt change
does".  Each iteration costs exactly one source-branch forward (no backprop), and the
identity ``eps_hat = (z - a * x0) / b`` means the same forward also gives ``e_j``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from .model import SDTurbo


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1).float()


def excess_kurtosis(x: torch.Tensor) -> float:
    v = _flat(x)
    mu = v.mean()
    var = ((v - mu) ** 2).mean()
    if float(var) <= 0.0:
        return float("nan")
    return float(((v - mu) ** 4).mean() / (var ** 2) - 3.0)


def eps_from_x0(z: torch.Tensor, x0: torch.Tensor, model: SDTurbo, t: float) -> torch.Tensor:
    """Recover the noise prediction from an x0 prediction (exact for eps-prediction)."""
    a, b = model.ab(t)
    return (z - a * x0) / b


def fixed_point_eps(
    model: SDTurbo,
    x_s: torch.Tensor,
    t: float,
    emb_src: torch.Tensor,
    eps0: torch.Tensor,
    j_max: int = 8,
) -> Tuple[List[torch.Tensor], List[Dict[str, float]]]:
    """``eps^(j+1) = eps_hat_src(a x_s + b eps^(j))``, ``j = 0 .. j_max``.

    Returns ``(eps_0 .. eps_{j_max}, diagnostics)``; diagnostics has one row per ``j`` with
    the source reconstruction error ``e_j``, the relative step to ``j+1`` (nan at ``j_max``),
    and the standard deviation / excess kurtosis of ``eps^(j)`` (is it still Gaussian?).
    """
    a, b = model.ab(t)
    eps = eps0
    x_s_norm = float(x_s.norm().item())
    eps_list: List[torch.Tensor] = []
    diag: List[Dict[str, float]] = []
    for j in range(int(j_max) + 1):
        z = a * x_s + b * eps
        x0 = model.x0_pred(z, t, emb_src)          # 1 UNet forward
        eps_hat = eps_from_x0(z, x0, model, t)
        e_j = float((x0 - x_s).norm().item()) / max(x_s_norm, 1e-12)
        step = float("nan")
        if j < int(j_max):
            step = float((eps_hat - eps).norm().item()) / max(float(eps.norm().item()), 1e-12)
        diag.append({
            "j": float(j),
            "e_src": e_j,
            "step": step,
            "eps_std": float(eps.float().std().item()),
            "eps_kurt": excess_kurtosis(eps),
        })
        eps_list.append(eps)
        eps = eps_hat
    return eps_list, diag


def antithetic(eps: torch.Tensor) -> torch.Tensor:
    """``-eps``: cancels the odd-order noise terms, so a 2-draw estimate is better centred."""
    return -eps


def select_draw(Rs: torch.Tensor, w: torch.Tensor) -> int:
    """``i* = argmin_i ||(1 - w) * R_i||^2`` -- the draw that disturbs the ignored region least.

    A model-internal criterion (§3.3 item "选样本而不是平均"): no CLIP, no GT mask.
    """
    keep = (1.0 - w).unsqueeze(0).unsqueeze(0)      # [1,1,64,64]
    energy = ((Rs * keep) ** 2).sum(dim=(1, 2, 3))  # [N]
    return int(torch.argmin(energy).item())


def mean_and_var(Rs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel mean and unbiased variance of a residual stack (used by P3b's report)."""
    n = Rs.shape[0]
    mean = Rs.mean(dim=0)
    var = ((Rs - mean[None]) ** 2).sum(dim=(0, 1)) / (4.0 * max(n - 1, 1))
    return mean, var


def share_of(Rs: Sequence[torch.Tensor], region: torch.Tensor) -> Dict[str, float]:
    """Fraction of the displacement energy falling inside ``region`` (diagnostic)."""
    stack = torch.stack([r[0] if r.dim() == 4 else r for r in Rs]).float()
    energy = (stack ** 2).mean(dim=1)               # [N,64,64]
    sel = region > 0.5
    total = float(energy.mean().item())
    inside = float(energy[:, sel].mean().item()) if bool(sel.any()) else float("nan")
    return {"energy_total": total, "energy_inside": inside,
            "share_inside": inside / total if total > 0 else float("nan")}
