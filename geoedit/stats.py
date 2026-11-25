"""Statistics: image-level bootstrap CIs, Wilcoxon signed-rank, matched-Pareto deltas, and
the v3 mean-level curve matching (§3.3).

Everything resamples **images** (the unit of independence).  Per-image interpolation is
deliberately *not* offered as a decision rule: P2 showed that a steep lambda curve turns it
into noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


# --------------------------------------------------------------------------- #
# v3: mean-level curve matching (§3.3)
# --------------------------------------------------------------------------- #
@dataclass
class CurveMatch:
    delta: float            # mean PSNR(name) - interp(mean PSNR(ref))(anchor CLIP)
    psnr_at_anchor: float
    anchor_clip: float
    out_of_range: bool
    n: int
    lo: float = float("nan")
    hi: float = float("nan")


def _interp_no_extrap(clip_curve: np.ndarray, y_curve: np.ndarray, clip_point: float):
    keep = np.isfinite(clip_curve) & np.isfinite(y_curve)
    clip, y = clip_curve[keep], y_curve[keep]
    if clip.size < 2 or not np.isfinite(clip_point):
        return None
    order = np.argsort(clip)
    clip, y = clip[order], y[order]
    if clip_point < clip.min() or clip_point > clip.max():
        return None
    return float(np.interp(clip_point, clip, y))


def mean_curve_match(
    curves: Dict[str, Dict[str, np.ndarray]],
    ref: str,
    anchor_index: Sequence[int],
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Match every variant's **mean** PSNR curve to the reference's at fixed CLIP anchors.

    ``curves[name] = {"clip": [L, n_img], "y": [L, n_img]}`` where ``L`` is the lambda grid.
    The anchor values are the reference's mean CLIP at the given grid indices, recomputed on
    every bootstrap resample of the image set (so the anchor itself is uncertain too).

    No extrapolation: a curve whose CLIP range does not contain the anchor is reported as
    ``out_of_range`` with ``delta = nan``.  Returns ``{"anchor_clip", "n", "results"}`` with
    ``results[lambda_index][name] = CurveMatch``.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    names = list(curves)
    n_img = curves[ref]["clip"].shape[1]
    boot: Dict[Tuple[int, str], List[float]] = {(int(i), name): [] for i in anchor_index for name in names}
    boot_abs: Dict[Tuple[int, str], List[float]] = {(int(i), name): [] for i in anchor_index for name in names}
    boot_anchor: Dict[int, List[float]] = {int(i): [] for i in anchor_index}

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n_img, n_img)
        ref_clip = np.nanmean(curves[ref]["clip"][:, idx], axis=1)
        for i in anchor_index:
            anchor = float(ref_clip[int(i)])
            boot_anchor[int(i)].append(anchor)
            ref_value = _interp_no_extrap(ref_clip, np.nanmean(curves[ref]["y"][:, idx], axis=1), anchor)
            if ref_value is None:
                continue
            for name in names:
                clip = np.nanmean(curves[name]["clip"][:, idx], axis=1)
                y = np.nanmean(curves[name]["y"][:, idx], axis=1)
                value = _interp_no_extrap(clip, y, anchor)
                if value is not None and ref_value is not None:
                    boot[(int(i), name)].append(float(value - ref_value))
                    boot_abs[(int(i), name)].append(float(value))

    full_clip = {name: np.nanmean(curves[name]["clip"], axis=1) for name in names}
    full_y = {name: np.nanmean(curves[name]["y"], axis=1) for name in names}
    results: Dict[str, Dict[str, CurveMatch]] = {}
    anchor_clip: Dict[str, float] = {}
    for i in anchor_index:
        i = int(i)
        anchor = float(full_clip[ref][i])
        anchor_clip[str(i)] = anchor
        column: Dict[str, CurveMatch] = {}
        ref_value = _interp_no_extrap(full_clip[ref], full_y[ref], anchor)
        for name in names:
            value = _interp_no_extrap(full_clip[name], full_y[name], anchor)
            deltas = np.asarray(boot[(i, name)], dtype=np.float64)
            if value is None or ref_value is None or deltas.size == 0:
                column[name] = CurveMatch(float("nan"), float("nan"), anchor, True, int(deltas.size))
                continue
            column[name] = CurveMatch(
                delta=float(value - ref_value),
                psnr_at_anchor=value,
                anchor_clip=anchor,
                out_of_range=False,
                n=int(deltas.size),
                lo=float(np.quantile(deltas, alpha / 2)),
                hi=float(np.quantile(deltas, 1 - alpha / 2)),
            )
        results[str(i)] = column
    return {
        "anchor_clip": anchor_clip,
        "n": int(n_img),
        "results": results,
        # raw image-level bootstrap draws, aligned across variants: paired differences
        # (H3c) and the gap-closure ratio G are computed from these
        "boot": {str(i): {name: boot[(i, name)] for name in names} for i in map(int, anchor_index)},
        "boot_abs": {str(i): {name: boot_abs[(i, name)] for name in names} for i in map(int, anchor_index)},
        "boot_anchor": {str(i): boot_anchor[i] for i in map(int, anchor_index)},
    }


def boot_mean_of_max(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """CI of ``max_l mean_i a[l,i] - max_l mean_i b[l,i]`` (H3a uses this on CLIP).

    Matrices are ``[L, n_img]``; NaNs are dropped per row, so the number of images can differ
    between lambdas.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    a = np.asarray(mat_a, dtype=np.float64)
    b = np.asarray(mat_b, dtype=np.float64)
    n_img = a.shape[1]

    def stat(mat: np.ndarray, idx: Optional[np.ndarray]) -> float:
        block = mat if idx is None else mat[:, idx]
        means = np.nanmean(block, axis=1)
        return float(np.nanmax(means))

    point = stat(a, None) - stat(b, None)
    boots = np.empty(int(n_boot), dtype=np.float64)
    for k in range(int(n_boot)):
        idx = rng.integers(0, n_img, n_img)
        boots[k] = stat(a, idx) - stat(b, idx)
    return {
        "point": point,
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(n_img),
    }


def _interp_loose(x: np.ndarray, y: np.ndarray, point: float):
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2 or not np.isfinite(point):
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    if point < x.min() or point > x.max():
        return None
    return float(np.interp(point, x, y))


def boot_psnr_at_clip(curves: Dict[str, Dict[str, np.ndarray]], names: Sequence[str],
                      clip_target: float, n_boot: int = 1000, alpha: float = 0.05,
                      rng: Optional[np.random.Generator] = None) -> Dict[str, Dict[str, float]]:
    """Background PSNR at a **fixed** CLIP value, image-level bootstrap, no extrapolation.

    Unlike :func:`mean_curve_match` the anchor is a number, not a lambda index, so it can be
    placed anywhere (used for the post-hoc anchors and for P3b's H3e).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n_img = curves[names[0]]["clip"].shape[1]
    draws: Dict[str, List[float]] = {name: [] for name in names}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n_img, n_img)
        for name in names:
            clip = np.nanmean(curves[name]["clip"][:, idx], axis=1)
            psnr = np.nanmean(curves[name]["y"][:, idx], axis=1)
            value = _interp_loose(clip, psnr, float(clip_target))
            if value is not None:
                draws[name].append(value)
    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        clip = np.nanmean(curves[name]["clip"], axis=1)
        psnr = np.nanmean(curves[name]["y"], axis=1)
        point = _interp_loose(clip, psnr, float(clip_target))
        values = np.asarray(draws[name], dtype=float)
        if point is None or values.size == 0:
            out[name] = {"psnr": float("nan"), "lo": float("nan"), "hi": float("nan"),
                         "out_of_range": True, "draws": values}
        else:
            out[name] = {"psnr": point, "lo": float(np.quantile(values, alpha / 2)),
                         "hi": float(np.quantile(values, 1 - alpha / 2)),
                         "out_of_range": False, "draws": values}
    return out


def boot_clip_at_psnr(curves: Dict[str, Dict[str, np.ndarray]], names: Sequence[str],
                      targets: Sequence[float], n_boot: int = 1000, alpha: float = 0.05,
                      rng: Optional[np.random.Generator] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """CLIP at a fixed background PSNR, image-level bootstrap (the other direction)."""
    rng = np.random.default_rng(0) if rng is None else rng
    n_img = curves[names[0]]["clip"].shape[1]
    draws = {(float(t), name): [] for t in targets for name in names}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n_img, n_img)
        for name in names:
            psnr = np.nanmean(curves[name]["y"][:, idx], axis=1)
            clip = np.nanmean(curves[name]["clip"][:, idx], axis=1)
            for target in targets:
                value = _interp_loose(psnr, clip, float(target))
                if value is not None:
                    draws[(float(target), name)].append(value)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for target in targets:
        out[f"{float(target):g}"] = {}
        for name in names:
            psnr = np.nanmean(curves[name]["y"], axis=1)
            clip = np.nanmean(curves[name]["clip"], axis=1)
            point = _interp_loose(psnr, clip, float(target))
            values = np.asarray(draws[(float(target), name)], dtype=float)
            if point is None or values.size == 0:
                out[f"{float(target):g}"][name] = {"clip": float("nan"), "lo": float("nan"),
                                                   "hi": float("nan"), "out_of_range": True}
            else:
                out[f"{float(target):g}"][name] = {
                    "clip": point, "lo": float(np.quantile(values, alpha / 2)),
                    "hi": float(np.quantile(values, 1 - alpha / 2)), "out_of_range": False}
    return out


def common_clip_range(curves: Dict[str, Dict[str, np.ndarray]], names: Sequence[str]):
    """Intersection of the mean-CLIP ranges of ``names`` (a bracket where all can be compared)."""
    los, his = [], []
    for name in names:
        clip = np.nanmean(curves[name]["clip"], axis=1)
        clip = clip[np.isfinite(clip)]
        if clip.size:
            los.append(float(clip.min()))
            his.append(float(clip.max()))
    if not los:
        return None
    return max(los), min(his)
