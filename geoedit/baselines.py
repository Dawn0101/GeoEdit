"""P11 baselines: ER-Chord, BudEdit, PreciseEdit (closed form on ChordEdit's 5-forward base)
plus the two SCAD refinements (adaptive alpha, clean-background output mask).

None of the three papers has runnable official code (BudEdit: "code coming soon"; the ER-Chord repo
link in arXiv 2610.29248 returns 404; PreciseEdit (2610.23535) publishes no repository), so each is
implemented from the paper's formula on the same base and at the same budget.  This is recorded as a
deviation: these are our reimplementations, not the authors' code.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .inlayer import plateau_mask


# ----------------------------------------------------------------------------------- #
# ER-Chord (arXiv 2610.29248): reliability re-allocation at equal energy
# ----------------------------------------------------------------------------------- #
def _smooth(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """3x3 box smoothing, one spatial plane per channel."""
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[None]
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)[0]


def er_chord_field(u_chord: torch.Tensor, R_h: torch.Tensor, R_l: torch.Tensor,
                   mu: float = 1.0, smooth: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """``w = 1/(1+mu (1-c)^2)`` on the per-pixel cross-scale cosine, then equal-energy rescale."""
    eps = 1e-8
    a, b = R_h[0].float(), R_l[0].float()
    if smooth > 1:
        a, b = _smooth(a, smooth), _smooth(b, smooth)
    num = (a * b).sum(dim=0)
    den = a.norm(dim=0) * b.norm(dim=0) + eps
    c = (num / den).clamp(0.0, 1.0)
    w = 1.0 / (1.0 + float(mu) * (1.0 - c) ** 2)
    uw = w.unsqueeze(0) * u_chord[0]
    scale = float(u_chord[0].norm()) / (float(uw.norm()) + eps)
    return (uw.unsqueeze(0) * scale), c


# ----------------------------------------------------------------------------------- #
# BudEdit (arXiv 2609.32841): attention-selected support + energy budget
# ----------------------------------------------------------------------------------- #
def attention_map(acc: list) -> torch.Tensor:
    """Peak-normalised mean of the captured attn2 edit-token maps (64x64)."""
    if not acc:
        return torch.ones(64, 64)
    m = torch.stack(acc, 0).mean(0)
    return m / m.max().clamp_min(1e-8)


def budedit_field(u_chord: torch.Tensor, A_l: torch.Tensor, phi: torch.Tensor,
                  beta: float = 0.5, max_gain: float = 2.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """``d = w_eff*(u + beta*A)`` with ``w_eff = G_attn * sqrt(phi*e)``, energy matched to ``||u||``."""
    eps = 1e-8
    g = (phi > phi.mean()).float()
    phi_pin = F.avg_pool2d(g[None, None], kernel_size=5, stride=1, padding=2,
                           count_include_pad=False)[0, 0]
    e = u_chord[0].norm(dim=0)
    e = e / e.max().clamp_min(eps)
    w_eff = g * (phi * e).clamp_min(0).sqrt()
    d_raw = w_eff.unsqueeze(0) * (u_chord[0] + float(beta) * A_l[0])
    gain = min(float(max_gain), float(u_chord[0].norm()) / (float(d_raw.norm()) + eps))
    return (d_raw * gain).unsqueeze(0), phi_pin


# ----------------------------------------------------------------------------------- #
# PreciseEdit (arXiv 2610.23535): SCT + directional gate + PreciseMask + composition
# ----------------------------------------------------------------------------------- #
def precise_mask(phi: torch.Tensor, attn1: torch.Tensor | None, beta_lap: float = 0.5,
                 T: float = 0.1) -> torch.Tensor:
    """Graph-Laplacian propagation of the attention seed, then an adaptive-threshold soft mask."""
    eps = 1e-8
    if attn1 is None:
        s = phi / phi.max().clamp_min(eps)
    else:
        g = int(round(attn1.shape[0] ** 0.5))
        m0 = F.interpolate(phi[None, None], size=(g, g), mode="bilinear",
                           align_corners=False)[0, 0].reshape(-1)
        prop = torch.mv(attn1.float(), m0.float())
        s32 = (1.0 - beta_lap) * m0 + beta_lap * prop
        s32 = s32.reshape(1, 1, g, g)
        s = F.interpolate(s32, size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
    s = (s - s.min()) / (s.max() - s.min() + eps)
    n = s.numel()
    e = float(s.sum()) ** 2 / (n * float((s ** 2).sum()) + eps)   # participation ratio
    rho = float(min(0.65, max(0.15, 0.35 + 0.5 * (e - 0.30))))
    tau = torch.quantile(s.flatten(), 1.0 - rho)
    m = torch.sigmoid((s - tau) / T)
    return (m - m.min()) / (m.max() - m.min() + eps)


def precise_compose(x_s: torch.Tensor, u_sct: torch.Tensor, M: torch.Tensor, gamma: float,
                    prox: torch.Tensor) -> torch.Tensor:
    """``x = x_s + gamma M^2 u_sct + M * (prox - x_tilde)`` per eq. 12-14 of the paper."""
    x_tilde = x_s + M.unsqueeze(0) * (float(gamma) * u_sct)
    r = prox - x_tilde
    return x_s + float(gamma) * (M ** 2).unsqueeze(0) * u_sct + M.unsqueeze(0) * r


def cosine_gate(u_h: torch.Tensor, u_l: torch.Tensor) -> float:
    a, b = u_h.float().flatten(), u_l.float().flatten()
    return max(0.0, float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-8)))


# ----------------------------------------------------------------------------------- #
# SCAD refinements
# ----------------------------------------------------------------------------------- #
def adaptive_alpha(r: float, alpha_max: float, r0: float) -> float:
    """``alpha = alpha_max * r0 / (r + r0)`` -- strong responses get a smaller boost."""
    return float(alpha_max) * float(r0) / (float(r) + float(r0) + 1e-8)


def two_masks(E: torch.Tensor, in_q: Tuple[float, float], out_q: Tuple[float, float]):
    """Wide mask for the layer guidance, tighter mask for the final composite."""
    return plateau_mask(E, *in_q), plateau_mask(E, *out_q)
