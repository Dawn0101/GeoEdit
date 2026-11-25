"""The edit operators: residual ``R``, one-step ``lam``, Picard iteration, ``avg``, ``prox``.

All of them share one noise tensor per ``(image_id, seed)`` and, per image, one cached
source-branch prediction ``mu_src = E[x0 | a x_s + b eps, c_src]``.  The source branch is
always queried at ``x_s`` (never at the current iterate), which is what makes Picard cost
``1 + K`` UNet forwards instead of ``2K``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .model import SDTurbo

LATENT_SHAPE = (4, 64, 64)


def image_seed(image_id: str, seed: int) -> int:
    """Deterministic 63-bit seed from ``(image_id, seed)`` (independent of dict order)."""
    digest = hashlib.sha256(f"{image_id}|{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def make_eps(
    image_id: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    shape: Sequence[int] = LATENT_SHAPE,
) -> torch.Tensor:
    """``[4, 64, 64]`` standard normal, reproducible across processes and machines.

    Generated on the CPU so the value does not depend on which GPU ran it.
    """
    generator = torch.Generator(device="cpu").manual_seed(image_seed(image_id, seed))
    eps = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    return eps.to(device=device, dtype=dtype)


@dataclass
class PicardResult:
    deltas: List[torch.Tensor]  # delta_1 .. delta_K  ([1,4,64,64] each)
    contraction: List[float]  # c_n for n >= 2 (len = K-1)
    norms: List[float]  # ||delta_n||
    diverged: bool
    nfe: int

    def delta(self, k: int) -> torch.Tensor:
        return self.deltas[k - 1]

    def diagnostics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for n, value in enumerate(self.norms, start=1):
            out[f"norm_{n}"] = value
        for n, value in enumerate(self.contraction, start=2):
            out[f"c_{n}"] = value
        out["diverged"] = float(self.diverged)
        return out


def noise_terms(model: SDTurbo, t: float) -> Tuple[float, float]:
    return model.ab(t)


def source_mu(
    model: SDTurbo,
    x_s: torch.Tensor,
    t: float,
    emb_src: torch.Tensor,
    eps: torch.Tensor,
) -> torch.Tensor:
    """``E[x0 | a x_s + b eps, c_src]`` -- one UNet forward."""
    a, b = noise_terms(model, t)
    z = a * x_s + b * eps
    return model.x0_pred(z, t, emb_src)


def residual_from_mu(
    model: SDTurbo,
    x: torch.Tensor,
    t: float,
    emb_tar: torch.Tensor,
    eps: torch.Tensor,
    mu_src: torch.Tensor,
) -> torch.Tensor:
    """``R(x) = E[x0 | a x + b eps, c_tar] - mu_src`` -- one UNet forward."""
    a, b = noise_terms(model, t)
    z = a * x + b * eps
    mu_tar = model.x0_pred(z, t, emb_tar)
    return mu_tar - mu_src


def residual(
    model: SDTurbo,
    x_s: torch.Tensor,
    t: float,
    emb_src: torch.Tensor,
    emb_tar: torch.Tensor,
    eps: torch.Tensor,
) -> torch.Tensor:
    """``R(x_s)`` -- two UNet forwards (source branch + target branch)."""
    mu_src = source_mu(model, x_s, t, emb_src, eps)
    return residual_from_mu(model, x_s, t, emb_tar, eps, mu_src)


def lam_edit(x_s: torch.Tensor, res: torch.Tensor, lam: float) -> torch.Tensor:
    return x_s + float(lam) * res


def picard(
    model: SDTurbo,
    x_s: torch.Tensor,
    t: float,
    emb_src: torch.Tensor,
    emb_tar: torch.Tensor,
    eps: torch.Tensor,
    k_max: int,
    divergence_ratio: float = 5.0,
    mu_src: Optional[torch.Tensor] = None,
) -> PicardResult:
    """``delta_0 = 0``, ``delta_n = R(x_s + delta_{n-1})``; returns every prefix.

    Costs ``1 + k_max`` UNet forwards.  Divergence is *flagged*, never truncated.
    """
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    if mu_src is None:
        mu_src = source_mu(model, x_s, t, emb_src, eps)

    deltas: List[torch.Tensor] = []
    norms: List[float] = []
    contraction: List[float] = []
    diverged = False
    consecutive_bad = 0

    delta = torch.zeros_like(x_s)
    for _ in range(int(k_max)):
        delta = residual_from_mu(model, x_s + delta, t, emb_tar, eps, mu_src)
        deltas.append(delta.detach().clone())
        norms.append(float(delta.norm().item()))

    # gaps[n] = ||delta_n - delta_{n-1}|| with delta_0 = 0; c_n = gaps[n] / gaps[n-1]
    gaps = [float("nan"), norms[0]]
    for n in range(2, len(deltas) + 1):
        gaps.append(float((deltas[n - 1] - deltas[n - 2]).norm().item()))
    for n in range(2, len(deltas) + 1):
        ratio = gaps[n] / gaps[n - 1] if gaps[n - 1] > 0 else float("inf")
        contraction.append(ratio)
        consecutive_bad = consecutive_bad + 1 if ratio > 1.0 else 0
        if consecutive_bad >= 2:
            diverged = True

    if norms and norms[-1] > divergence_ratio * norms[0]:
        diverged = True

    return PicardResult(
        deltas=deltas,
        contraction=contraction,
        norms=norms,
        diverged=diverged,
        nfe=1 + int(k_max),
    )


def avg_residual(
    model: SDTurbo,
    x_s: torch.Tensor,
    t: float,
    emb_src: torch.Tensor,
    emb_tar: torch.Tensor,
    eps_list: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Mean of ``R(x_s)`` over independent noise draws (variance-reduction control)."""
    if len(eps_list) == 0:
        raise ValueError("eps_list is empty")
    total = None
    for eps in eps_list:
        res = residual(model, x_s, t, emb_src, emb_tar, eps)
        total = res if total is None else total + res
    return total / float(len(eps_list))


def prox(
    model: SDTurbo,
    x: torch.Tensor,
    t_c: float,
    emb_tar: torch.Tensor,
    eps_c: torch.Tensor,
) -> torch.Tensor:
    """One Tweedie projection towards the target prompt at ``t_c`` (1 UNet forward)."""
    a, b = noise_terms(model, t_c)
    z = a * x + b * eps_c
    return model.x0_pred(z, t_c, emb_tar)


def masked_norms(
    delta: torch.Tensor,
    mask64: torch.Tensor,
    dist64: torch.Tensor,
    band: Tuple[float, float] = (2.0, 6.0),
) -> Dict[str, float]:
    """Mean squared displacement inside / outside / in a distance band (latent units)."""
    sq = (delta[0].float() ** 2).mean(dim=0)  # [64,64], mean over channels
    interior = (mask64 > 0.5).float()
    outside = 1.0 - interior
    band_mask = (dist64 > band[0]) & (dist64 <= band[1])
    interior_frac = interior.mean().clamp_min(1e-6)
    outside_frac = outside.mean().clamp_min(1e-6)
    return {
        "delta_in": float((sq * interior).sum() / interior_frac),
        "delta_out": float((sq * outside).sum() / outside_frac),
        "delta_ring": float(sq[band_mask].mean()) if band_mask.any() else float("nan"),
    }
