"""Statistics: image-level bootstrap CIs, Wilcoxon signed-rank, and matched-Pareto deltas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats


def _clean(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def boot_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    statistic: Callable[[np.ndarray], float] = np.median,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Percentile bootstrap CI resampling *images* (the unit of independence)."""
    rng = np.random.default_rng(0) if rng is None else rng
    clean = _clean(values)
    if clean.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        sample = clean[rng.integers(0, clean.size, clean.size)]
        boots[i] = statistic(sample)
    return {
        "point": float(statistic(clean)),
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(clean.size),
    }


def paired_boot_ci(
    diffs: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Bootstrap CI of the mean of paired differences."""
    rng = np.random.default_rng(0) if rng is None else rng
    clean = _clean(diffs)
    if clean.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    boots = np.array([clean[rng.integers(0, clean.size, clean.size)].mean()
                      for _ in range(int(n_boot))])
    return {
        "point": float(clean.mean()),
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(clean.size),
    }


def wilcoxon_p(values: Sequence[float], alternative: str = "less") -> Dict[str, float]:
    """One-sided Wilcoxon signed-rank test against 0 (zeros dropped by scipy)."""
    clean = _clean(values)
    if clean.size < 10:
        return {"stat": float("nan"), "p": float("nan"), "n": int(clean.size)}
    result = scipy_stats.wilcoxon(clean, alternative=alternative, zero_method="wilcox")
    return {"stat": float(result.statistic), "p": float(result.pvalue), "n": int(clean.size)}


@dataclass
class MatchedDelta:
    delta: float          # psnr_point - interp(curve)(clip_point), nan when out of range
    psnr_curve_at: float  # interp(curve)(clip_point)
    clip_point: float
    out_of_range: bool
    n: int


def interp_curve(clip_curve: Sequence[float], y_curve: Sequence[float], clip_point: float) -> MatchedDelta:
    """Linear interpolation of a reference curve at ``clip_point``, no extrapolation."""
    clip = np.asarray(clip_curve, dtype=np.float64)
    y = np.asarray(y_curve, dtype=np.float64)
    keep = np.isfinite(clip) & np.isfinite(y)
    clip, y = clip[keep], y[keep]
    if clip.size < 2 or not np.isfinite(clip_point):
        return MatchedDelta(float("nan"), float("nan"), float(clip_point), True, int(clip.size))
    order = np.argsort(clip)
    clip, y = clip[order], y[order]
    if clip_point < clip.min() or clip_point > clip.max():
        return MatchedDelta(float("nan"), float("nan"), float(clip_point), True, int(clip.size))
    value = float(np.interp(clip_point, clip, y))
    return MatchedDelta(float("nan"), value, float(clip_point), False, int(clip.size))


def delta_vs_curve(
    clip_curve: Sequence[float],
    y_curve: Sequence[float],
    clip_point: float,
    y_point: float,
) -> MatchedDelta:
    """``y_point - interp(curve)(clip_point)``; ``nan`` and flagged when out of range."""
    matched = interp_curve(clip_curve, y_curve, clip_point)
    if matched.out_of_range:
        return matched
    return MatchedDelta(float(y_point - matched.psnr_curve_at), matched.psnr_curve_at,
                        float(clip_point), False, matched.n)
