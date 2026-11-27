"""P6: the Ours-4 update --- DDS residual plus an in-mask SDS/attraction term.

The method, at exactly **4 UNet forwards per image**:

* ``t_l = 0.5``, seed 0 (2 forwards)  -> ``R0(0.5)``: its energy ``||R0(0.5)||`` smoothed with a
  Gaussian of ``sigma = 2`` is the mask ``M`` (P5 §2: median AUROC 0.8565, equal to the 8-NFE
  four-seed average of 0.8581);
* ``t_h = 0.9``, seed 0 (2 forwards)  -> ``R0(0.9) = mu_tar - mu_src`` **and** the attraction
  ``A0(0.9) = mu_tar - x_s``, which costs nothing extra because ``mu_tar`` is already computed.

The composed update is

    x = x_s + lam * M * (R + k * M**(gamma-1) * A)

with ``k = 0`` recovering the plain masked DDS transport (P3's V12 with a smoothed mask) and
``gamma > 1`` concentrating the extra push on the mask core (``M**(gamma-1)`` is 1 everywhere on
the support when ``gamma = 1``).

**Why ``A`` is not just ``R``.**  ``A = R + e_src`` with ``e_src = mu_src - x_s`` the source-side
prediction error at ``t_h``.  ``R`` is a difference of two branch predictions, so ``e_src`` is
subtracted; ``A`` adds it back.  ``e_src = 0`` only under the one-step-edit assumption that the
UNet's eps prediction equals the injected noise, which is an approximation, not an identity --
``scripts/61_p6_sds_diagnose.py`` measures it.  Note also that this term has the same form as
PreciseEdit's anchor-relative target attraction ``k * (mu_tar - x)`` (arXiv 2610.23535, eq. 5-6);
our contribution is the mask and the NFE budget, not the attraction.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from .masks import gaussian_blur

SIGMA_4NFE = 2.0  # the L5 smoother selected in P5 §2
LATENT_KEYS = ("t0.9_s0", "t0.5_s0")


def l5_mask(R5_single: torch.Tensor, sigma: float = SIGMA_4NFE) -> torch.Tensor:
    """``[4,64,64]`` single-seed residual at t_l -> ``[64,64]`` mask in [0,1], max-normalised."""
    energy = R5_single.norm(dim=0) if R5_single.dim() == 3 else R5_single
    smooth = gaussian_blur(energy.float(), sigma)
    return smooth / smooth.max().clamp_min(1e-12)


def attention_term(R9: torch.Tensor, mu_src9: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
    """``A = mu_tar - x_s``; ``R9`` must be ``mu_tar - mu_src9`` (the cached residual)."""
    return R9 + (mu_src9 - x_s)


def ours4_delta(R9: torch.Tensor, A9: torch.Tensor, M: torch.Tensor, lam: float, k: float,
                gamma: float = 1.0) -> torch.Tensor:
    """``[4,64,64]`` displacement ``lam * M * (R + k * M**(gamma-1) * A)``."""
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1 (M**(gamma-1) is the core-concentration exponent)")
    m = M.float()
    core = m ** (gamma - 1.0)  # gamma=1 -> 1 everywhere on the support; gamma=2 -> M
    return float(lam) * m * (R9.float() + float(k) * core * A9.float())


def own_norm(delta: torch.Tensor) -> float:
    return float(delta.float().norm())


def describe(R9, A9, M, k: float, gamma: float) -> Dict[str, float]:
    """Diagnostics for one draw: how much of ``A`` is ``R``, and where the push lands."""
    m = M.float()
    a_norm = float(A9.norm())
    r_norm = float(R9.norm())
    cos = float((R9 * A9).sum() / (r_norm * a_norm + 1e-12))
    sel = m > 0.5
    if bool(sel.any()):
        r_in = float(R9[:, sel].norm() / sel.sum().sqrt())
        a_in = float(A9[:, sel].norm() / sel.sum().sqrt())
    else:
        r_in = a_in = float("nan")
    return {"R_norm": r_norm, "A_norm": a_norm, "cos_RA": cos,
            "rms_R_in_mask": r_in, "rms_A_in_mask": a_in,
            "mask_area_frac": float(sel.float().mean()), "k": float(k), "gamma": float(gamma)}


# --------------------------------------------------------------------------------------- #
# P7: Ours-5 -- dual-scale fusion, direction gating, energy-normalised core, masked prox
# --------------------------------------------------------------------------------------- #
def scale_match(u_h: torch.Tensor, u_l: torch.Tensor) -> torch.Tensor:
    """Low-noise field rescaled to the high-noise field's norm (so fusion is about *direction*)."""
    return u_l.float() * (u_h.float().norm() / u_l.float().norm().clamp_min(1e-12))


def fuse_scales(u_h: torch.Tensor, u_l: torch.Tensor, w: float) -> torch.Tensor:
    """``u = u_h + w * s * u_l`` with ``s = ||u_h|| / ||u_l||`` (``w=0`` disables the fusion)."""
    if w <= 0.0:
        return u_h.float()
    return u_h.float() + float(w) * scale_match(u_h, u_l)


def direction_gate(u_h: torch.Tensor, u_l: torch.Tensor) -> float:
    """PreciseEdit eq. 9: ``max(0, cos(u_h, u_l))``, a per-image scalar (0 extra NFE)."""
    a = u_h.float().flatten()
    b = u_l.float().flatten()
    cos = float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))
    return max(0.0, cos)


def energy_normalise(delta: torch.Tensor, reference: torch.Tensor, lam: float) -> torch.Tensor:
    """Rescale ``delta`` so that ``||delta|| == lam * ||reference||``.

    This is the fix for the ``k>0`` blow-up: the SDS term has ``||e_src|| = 1.77 ||R||``, so
    without it the step length grows with k and the image is destroyed.  After normalisation the
    lambda grid controls the amplitude identically for every variant, so a matched-PSNR comparison
    differs only in *direction and spatial profile*.
    """
    target = float(lam) * float(reference.float().norm())
    return delta.float() * (target / delta.float().norm().clamp_min(1e-12))


def ours5_delta(u: torch.Tensor, A_h: torch.Tensor, M: torch.Tensor, lam: float, k: float,
                gamma: float = 2.0, reference: torch.Tensor | None = None) -> torch.Tensor:
    """``lam * M * (u + k * M**gamma * A_h)``, optionally renormalised to ``lam * ||reference||``.

    ``gamma`` is the exponent on the mask for the *attraction* term only (``gamma=2`` means the
    SDS push is confined to the mask core), matching the P7 spec in ``configs/p7.yaml``.
    """
    m = M.float()
    delta = m * (u.float() + float(k) * (m ** float(gamma)) * A_h.float())
    delta = float(lam) * delta
    if reference is not None:
        return energy_normalise(delta, reference, lam)
    return delta


def masked_prox(x_s: torch.Tensor, y_edit: torch.Tensor, prox: torch.Tensor,
                M: torch.Tensor) -> torch.Tensor:
    """``(1 - M) * x_s + M * prox`` -- the 5th forward's spatially gated composite."""
    m = M.float()
    return (1.0 - m) * x_s.float() + m * prox.float()


def prox_latent(model, y_edit: torch.Tensor, t_c: float, emb_tar: torch.Tensor,
                eps: torch.Tensor) -> torch.Tensor:
    """One UNet forward at ``t_c`` on the *target* branch, same noise realisation as the transport.

    Identical to ChordEdit's ``prox`` / ``_pred_x0`` (verified: both use
    ``timestep = round(t * max_timestep)`` and ``alpha = sqrt(abar)``), so Ours-5 and the
    5-forward ChordEdit baseline are compared at the same NFE *and* the same prox recipe.
    """
    squeeze = y_edit.dim() == 3
    y = y_edit.float().unsqueeze(0) if squeeze else y_edit.float()
    e = eps.float().unsqueeze(0) if eps.dim() == 3 else eps.float()
    a, b = model.ab(t_c)
    z = float(a) * y + float(b) * e
    out = model.x0_pred(z, t_c, emb_tar)
    return out[0] if squeeze else out
