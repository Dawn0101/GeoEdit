"""Per-pixel masks from several noise draws (P3 §2.1).

Everything here works on *residual stacks* ``Rs`` of shape ``[N, 4, 64, 64]`` (N >= 2 draws,
same ``t``, same image) and returns a ``[64, 64]`` weight in ``[0, 1]``.

The main construction is the noise-calibrated SNR weight

    T_p = N * mean(||m_p||^2) / (4 * mean(s_p^2) + 1e-12),    w_p = max(0, 1 - c / T_p)

with ``m`` the per-pixel mean over the N draws and ``s_p^2`` the unbiased per-pixel variance
(``1 / (4(N-1)) * sum_i sum_c (R_icp - m_cp)^2``).  The ``N`` and the ``4`` make ``T``
*unbiased at 1* when there is no signal (T-K1), so ``w = 1 - 1/T`` with ``c = 1`` is a
James-Stein style positive-part shrinkage whose threshold is the noise level itself: no
hand-tuned threshold, and no CLIP anywhere.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _box(x: torch.Tensor, pool: int) -> torch.Tensor:
    """3x3 box filter with stride 1 (border = mean of the valid elements only)."""
    if pool <= 1:
        return x
    return F.avg_pool2d(x[None, None], kernel_size=pool, stride=1, padding=pool // 2,
                        count_include_pad=False)[0, 0]


def stack_residuals(residuals) -> torch.Tensor:
    """List of ``[1,4,64,64]`` or ``[4,64,64]`` -> ``[N,4,64,64]`` fp32."""
    rows = [r[0] if r.dim() == 4 else r for r in residuals]
    return torch.stack([r.float() for r in rows], dim=0)


def moments(Rs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel mean ``m`` ``[4,64,64]`` and unbiased variance ``s2`` ``[64,64]``."""
    n = Rs.shape[0]
    if n < 2:
        raise ValueError("need N >= 2 noise draws to estimate a variance")
    mean = Rs.mean(dim=0)
    s2 = ((Rs - mean[None]) ** 2).sum(dim=(0, 1)) / (4.0 * (n - 1))
    return mean, s2


def snr_weight(Rs: torch.Tensor, c: float = 1.0, pool: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Noise-calibrated weight ``w`` and the raw SNR map ``T`` (§2.1 steps 1-4)."""
    n = Rs.shape[0]
    mean, s2 = moments(Rs)
    m_sq = _box((mean ** 2).sum(dim=0), pool)          # smoothed ||m_p||^2
    s_sq = _box(s2, pool)                              # smoothed s_p^2
    T = (n * m_sq) / (4.0 * s_sq + 1e-12)
    w = torch.clamp(1.0 - float(c) / T.clamp_min(1e-12), min=0.0, max=1.0)
    return w, T


def diffedit_mask(Rs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """DiffEdit-style: mean ``||R_i,p||`` over draws, divided by the map max, then threshold."""
    energy = Rs.norm(dim=1).mean(dim=0)                # [64,64], ||.|| over 4 channels
    return (energy / energy.max().clamp_min(1e-12) > threshold).float()


def energy_mask(R: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Single-draw energy support (BudEdit's energy support without the attention term)."""
    energy = R[0].norm(dim=0) if R.dim() == 4 else R.norm(dim=0)
    return (energy / energy.max().clamp_min(1e-12) > threshold).float()


def variance_decomp(Rs: torch.Tensor, region: torch.Tensor) -> Dict[str, float]:
    """Random vs systematic energy inside a region (§3.2 item 1), in latent units.

    ``V = mean_p(4 s_p^2)`` estimates the per-pixel random energy; ``S`` is the unbiased
    estimate of the systematic part ``mean_p(||mu_p||^2 - 4 s_p^2 / N)``.  Both can be
    noisy for a single image; the report only uses the median over images.
    """
    n = Rs.shape[0]
    mean, s2 = moments(Rs)
    m_sq = (mean ** 2).sum(dim=0)
    sel = region > 0.5
    if not bool(sel.any()):
        return {"V": float("nan"), "S": float("nan"), "frac_random": float("nan")}
    V = float((4.0 * s2[sel]).mean())
    S = float((m_sq[sel] - 4.0 * s2[sel] / n).mean())
    total = S + V
    return {"V": V, "S": S, "frac_random": V / total if total > 0 else float("nan")}


# --------------------------------------------------------------------------- #
# Mask quality
# --------------------------------------------------------------------------- #
def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # average ranks for ties
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(counts) != len(x):
        sums = np.zeros(len(counts), dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def auroc(scores, gt) -> float:
    """AUROC of a continuous ``scores`` map against a binary ``gt`` map (ties averaged)."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    g = (np.asarray(gt).ravel() > 0.5)
    n_pos, n_neg = int(g.sum()), int((~g).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank(s)
    return float((ranks[g].sum() - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg))


def iou(mask, gt) -> float:
    """IoU of two binary maps (a continuous mask is thresholded at 0.5)."""
    m = np.asarray(mask) > 0.5
    g = np.asarray(gt) > 0.5
    union = int((m | g).sum())
    if union == 0:
        return float("nan")
    return float((m & g).sum()) / union


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()
