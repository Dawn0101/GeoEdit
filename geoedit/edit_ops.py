"""The edit operators: noise, residual ``R``, one-step ``lam``, and the noise average.

Every operator shares one noise tensor per ``(image_id, seed)``.  ``R`` costs two UNet
forwards (source branch + target branch); the mean over several noises (``avg_residual``)
is v3's variant V1 and P2's variance-reduction control.

v2's ``picard`` / ``prox`` / ``PicardResult`` live on the ``v2`` branch: P2 refuted both for
the main analysis and v3 only uses raw edits (``x = x_s + lambda * delta``).
"""

from __future__ import annotations

import hashlib
from typing import Dict, Sequence, Tuple

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


def avg_residual(
    model: SDTurbo,
    x_s: torch.Tensor,
    t: float,
    emb_src: torch.Tensor,
    emb_tar: torch.Tensor,
    eps_list: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Mean of ``R(x_s)`` over independent noise draws (v3's variant V1)."""
    if len(eps_list) == 0:
        raise ValueError("eps_list is empty")
    total: torch.Tensor | None = None
    for eps in eps_list:
        res = residual(model, x_s, t, emb_src, emb_tar, eps)
        total = res if total is None else total + res
    assert total is not None
    return total / float(len(eps_list))


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
    band_any = bool(band_mask.any())
    return {
        "delta_in": float((sq * interior).sum() / interior_frac),
        "delta_out": float((sq * outside).sum() / outside_frac),
        "delta_ring": float(sq[band_mask].mean()) if band_any else float("nan"),
    }
