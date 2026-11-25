"""Halo statistics: distance bands, the blurred-interior reference ``u``, and ``rho``.

``u = blur_sigma(R * I)`` is the residual smoothed from the interior outwards.  If a
one-step edit leaves a counter-typed ring just outside the object, the residual in the
ring is anti-correlated with the interior residual that bled into it, so
``rho_band = cos(R|band, u|band) < 0``.  A mask-shift null tells us how much positive
correlation the residual's own spatial smoothness produces by accident.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

BANDS: Tuple[Tuple[float, float], ...] = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 10.0), (10.0, np.inf))
MAIN_BAND = (2.0, 6.0)


def band_mask(dist64: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return (dist64 > lo) & (dist64 <= hi)


def band_masks(dist64: np.ndarray, bands: Iterable[Tuple[float, float]] = BANDS) -> Dict[str, np.ndarray]:
    return {f"{lo:g}_{hi:g}" if np.isfinite(hi) else f"{lo:g}_inf": band_mask(dist64, (lo, hi))
            for lo, hi in bands}


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur, independent per channel. ``x``: ``[B, C, H, W]``."""
    if sigma <= 0:
        return x
    radius = int(3 * sigma)
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    kx = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    pad = (radius, radius, radius, radius)
    out = F.conv2d(F.pad(x, pad, mode="replicate"), kx, groups=channels)
    return F.conv2d(out, ky, groups=channels)


def _flatten_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select the masked pixels of ``[B, C, H, W]`` -> ``[m * C]`` (channels interleaved)."""
    if mask.dim() == 4:
        mask = mask[:, 0]
    return x.permute(0, 2, 3, 1)[mask].reshape(-1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item()) * float(b.norm().item())
    if denom <= 0:
        return float("nan")
    return float((a * b).sum().item() / denom)


def rho(
    res: torch.Tensor,
    mask64: np.ndarray,
    dist64: np.ndarray,
    band: Tuple[float, float] = MAIN_BAND,
    sigma: float = 3.0,
) -> Tuple[float, int]:
    """``cos(R|band, blur_sigma(R * I)|band)`` and the number of pixels in the band.

    ``res``: ``[1, 4, 64, 64]``; ``mask64``: ``[1, 1, 64, 64]`` in ``{0, 1}``.
    Returns ``(nan, n)`` when the band is empty or the reference is degenerate.
    """
    res = res.float()
    mask_t = torch.from_numpy(np.asarray(mask64, dtype=np.float32))[None, None].to(res.device)
    u = gaussian_blur(res * mask_t, sigma)
    pixels = torch.from_numpy(band_mask(np.asarray(dist64), band)).to(res.device)[None]
    n_pix = int(pixels.sum().item())
    if n_pix == 0:
        return float("nan"), 0
    return cosine(_flatten_masked(res, pixels), _flatten_masked(u, pixels)), n_pix


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    h, w = mask.shape
    src_y = slice(max(0, -dy), min(h, h - dy))
    src_x = slice(max(0, -dx), min(w, w - dx))
    dst_y = slice(max(0, dy), min(h, h + dy))
    dst_x = slice(max(0, dx), min(w, w + dx))
    shifted[dst_y, dst_x] = mask[src_y, src_x]
    return shifted


def legal_shifts(mask64: np.ndarray, gap: float = 6.0) -> np.ndarray:
    """Translations whose interior stays in frame and clears the mask dilated by ``gap``.

    ``gap`` is Euclidean (a disk), which is the literal reading of "6 latent cells away"
    and leaves the corners available, unlike a square dilation.  Returns ``[n, 2]``
    ``(dy, dx)`` pairs; empty when no such translation exists (large objects).
    """
    interior = np.asarray(mask64) > 0.5
    if not interior.any():
        return np.zeros((0, 2), dtype=np.int32)
    ys, xs = np.nonzero(interior)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    h, w = interior.shape
    forbidden = ndimage.distance_transform_edt(~interior) <= float(gap)

    legal = []
    for dy in range(-y0, h - y1):
        for dx in range(-x0, w - x1):
            shifted = _shift_mask(interior, dy, dx)
            if not (shifted & forbidden).any():
                legal.append((dy, dx))
    if not legal:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(legal, dtype=np.int32)


def rho_null(
    res: torch.Tensor,
    mask64: np.ndarray,
    n_shifts: int = 20,
    gap: float = 6.0,
    band: Tuple[float, float] = MAIN_BAND,
    sigma: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, int]:
    """Mean ``rho`` over ``n_shifts`` random legal mask translations (``nan`` if none)."""
    rng = np.random.default_rng(0) if rng is None else rng
    mask64 = np.asarray(mask64)
    interior = mask64 > 0.5
    if not interior.any():
        return float("nan"), 0

    shifts = legal_shifts(mask64, gap=gap)
    if len(shifts) == 0:
        return float("nan"), 0
    take = min(int(n_shifts), len(shifts))
    chosen = shifts[rng.choice(len(shifts), size=take, replace=False)]

    values = []
    for dy, dx in chosen:
        shifted = _shift_mask(interior, int(dy), int(dx))
        dist_shifted = ndimage.distance_transform_edt(~shifted).astype(np.float32)
        value, n_pix = rho(res, shifted.astype(np.float32), dist_shifted, band=band, sigma=sigma)
        if n_pix > 0 and np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), 0
    return float(np.mean(values)), len(values)


def e_rel(
    res: torch.Tensor,
    mask64: np.ndarray,
    dist64: np.ndarray,
    band: Tuple[float, float] = MAIN_BAND,
) -> float:
    """``||R|_band||^2 / ||R|_I||^2``, both as per-pixel mean squares."""
    res = res.float()
    sq = (res[0] ** 2).mean(dim=0)
    interior = torch.from_numpy(np.asarray(mask64) > 0.5).to(res.device)
    pixels = torch.from_numpy(band_mask(np.asarray(dist64), band)).to(res.device)
    if not bool(interior.any()) or not bool(pixels.any()):
        return float("nan")
    return float(sq[pixels].mean().item() / sq[interior].mean().clamp_min(1e-12).item())


def prepare_null(
    mask64: np.ndarray,
    n_shifts: int = 20,
    gap: float = 6.0,
    rng: Optional[np.random.Generator] = None,
) -> list:
    """Precompute the null translations once per image: ``[(shifted_mask, dist), ...]``.

    The legal shifts depend only on the mask, so all (t, seed) of an image reuse them;
    recomputing inside ``rho_null`` would cost 6x more distance transforms.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    interior = np.asarray(mask64) > 0.5
    if not interior.any():
        return []
    shifts = legal_shifts(mask64, gap=gap)
    if len(shifts) == 0:
        return []
    take = min(int(n_shifts), len(shifts))
    chosen = shifts[rng.choice(len(shifts), size=take, replace=False)]
    prepared = []
    for dy, dx in chosen:
        shifted = _shift_mask(interior, int(dy), int(dx)).astype(np.float32)
        prepared.append((shifted, ndimage.distance_transform_edt(shifted < 0.5).astype(np.float32)))
    return prepared


def rho_null_prepared(
    res: torch.Tensor,
    prepared: list,
    band: Tuple[float, float] = MAIN_BAND,
    sigma: float = 3.0,
) -> Tuple[float, int]:
    """Mean ``rho`` over precomputed null translations."""
    values = []
    for shifted, dist in prepared:
        value, n_pix = rho(res, shifted, dist, band=band, sigma=sigma)
        if n_pix > 0 and np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), 0
    return float(np.mean(values)), len(values)
