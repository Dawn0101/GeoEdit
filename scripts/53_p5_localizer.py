"""P5-§2: can a **single** noise draw be made competitive by spatial smoothing?

P4 §5 / P3a established that the mean energy over 4 seeds localises the edit (AUROC 0.858 at
t=0.5) while a single draw only reaches ~0.79.  If the *random* part of the residual were spatially
white, replacing "average over 4 seeds" with "smooth one seed" would be free.  This script tests
both halves of that statement offline from the P1 residuals -- no UNet forward.

Localisers (AUROC of the continuous map against ``mask64``, 540 images, categories 0-7):

``L0``   ``||R0(0.5)||``                        single seed, no smoothing          2 NFE
``L1``   pool3 of L0                            3x3 box, stride 1                  2 NFE
``L2``   pool5 of L0                            5x5 box, stride 1                  2 NFE
``L3-L5`` Gaussian of L0, sigma = 1 / 1.5 / 2                                      2 NFE
``L6``   sqrt(pool3 L0(0.5) * pool3 L0(0.9))    geometric mean of two scales        4 NFE
``L7``   mean energy over 4 seeds (t=0.5)       == the V12 / DiffEdit mask           8 NFE

Spatial autocorrelation of the **random part**: ``D_i = R_i - mean(R)`` over the 4 seeds, its
per-pixel energy ``e_i = ||D_i||_2``, mean-removed, correlated at lags 1, 2, 3, 5 px, reported
inside and outside ``mask64``.  A white field gives ~0; the higher it is, the less a spatial filter
can substitute for averaging.

Outputs ``results/p5/localizer/*.csv`` + ``p5_localizer_verdict.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402
from geoedit.masks import auroc, gaussian_blur  # noqa: E402

LAGS = (1, 2, 3, 5)


def box(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(x[None, None], kernel_size=k, stride=1, padding=k // 2,
                        count_include_pad=False)[0, 0]


gauss = gaussian_blur


def localisers(R5: torch.Tensor, R9: torch.Tensor) -> Dict[str, torch.Tensor]:
    """``R5``/``R9``: ``[4,4,64,64]`` residuals at t=0.5 / 0.9 for seeds 0-3."""
    e5 = R5[0].norm(dim=0)
    e9 = R9[0].norm(dim=0)
    out = {
        "L0": e5,
        "L1": box(e5, 3),
        "L2": box(e5, 5),
        "L3": gauss(e5, 1.0),
        "L4": gauss(e5, 1.5),
        "L5": gauss(e5, 2.0),
        "L6": torch.sqrt(box(e5, 3).clamp_min(0) * box(e9, 3).clamp_min(0)),
        "L7": R5.norm(dim=1).mean(dim=0),
    }
    return out


def autocorr_energy(D: torch.Tensor, region: np.ndarray) -> Dict[str, float]:
    """Mean-removed autocorrelation of the per-pixel energy of the random part ``D`` [N,4,64,64].

    Reported separately **inside** (``ac{k}``) and **outside** (``ac{k}_out``) the mask: the
    residual energy also tracks image content (edges), so a mixture of the two regions would
    confound spatial dependence of the *noise* with spatial dependence of the *scene*.
    """
    e = D.norm(dim=1).numpy()  # [N,64,64]
    num = {"in": {}, "out": {}}
    den = {"in": {}, "out": {}}
    for lag in LAGS:
        num["in"][lag] = num["out"][lag] = 0.0
        den["in"][lag] = den["out"][lag] = 0.0
        for i in range(e.shape[0]):
            f = e[i]
            for axis in (0, 1):
                a = f if axis == 0 else f.T
                b = np.roll(a, lag, axis=0)
                sel = np.ones_like(a, dtype=bool)
                sel[-lag:, :] = False  # drop the wrapped rows
                for mask_sel, tag in ((region.astype(bool), "in"),
                                      ((~region.astype(bool)), "out")):
                    s = sel & (mask_sel if axis == 0 else mask_sel.T)
                    if s.sum() < 50:
                        continue
                    va, vb = a[s], b[s]
                    va = va - va.mean()
                    vb = vb - vb.mean()
                    num[tag][lag] += float((va * vb).sum())
                    den[tag][lag] += float(np.sqrt((va ** 2).sum() * (vb ** 2).sum()))
    out: Dict[str, float] = {}
    for lag in LAGS:
        out[f"ac{lag}"] = num["in"][lag] / den["in"][lag] if den["in"][lag] > 0 else float("nan")
        out[f"ac{lag}_out"] = num["out"][lag] / den["out"][lag] if den["out"][lag] > 0 else float("nan")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = [i for i in range(len(dataset)) if dataset[i].category not in ("8", "9")]
    if args.limit:
        pool = pool[: args.limit]
    a_root = ROOT / base["output_root"] / "p3" / "a"
    out_dir = ROOT / base["output_root"] / "p5" / "localizer"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    ac_rows: List[dict] = []
    for n, index in enumerate(pool):
        sample = dataset[index]
        bundle = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
        R5 = torch.stack([bundle[f"t0.5_s{s}"] for s in range(4)], 0).float()
        R9 = torch.stack([bundle[f"t0.9_s{s}"] for s in range(4)], 0).float()
        gt = sample.mask64
        for name, m in localisers(R5, R9).items():
            rows.append({"image_id": sample.image_id, "category": sample.category,
                         "method": name, "auroc": auroc(m.numpy(), gt),
                         "area": float(gt.mean())})
        # random part = residual minus its per-pixel mean over the 4 seeds
        D5 = R5 - R5.mean(dim=0, keepdim=True)
        ac = autocorr_energy(D5, gt.astype(bool))
        ac.update({"image_id": sample.image_id, "category": sample.category,
                   "frac_random": 0.0})
        ac_rows.append(ac)
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(pool)}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "p5_localizer_auroc.csv", index=False)
    ac = pd.DataFrame(ac_rows)
    ac.to_csv(out_dir / "p5_localizer_autocorr.csv", index=False)
    summary = (frame.groupby("method")["auroc"]
               .agg(["mean", "median", "std", "count"]).reindex(
                   ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]).round(4))
    summary.to_csv(out_dir / "p5_localizer_summary.csv")
    print(summary.to_string())

    ac_summary = {}
    for l in LAGS:
        for tag, name in (("", "inside"), ("_out", "outside")):
            ac_summary[f"ac{l}{tag}"] = {
                "region": name, "mean": float(ac[f"ac{l}{tag}"].mean()),
                "median": float(ac[f"ac{l}{tag}"].median())}
    print("\nautocorrelation of the random part's energy (t=0.5):")
    for k, v in ac_summary.items():
        print(f"  lag {k[2:5]:>2s} px {v['region']:7s} mean {v['mean']:+.4f}  "
              f"median {v['median']:+.4f}")

    # per-category AUROC, to show the ranking is not driven by one category
    per_cat = (frame.groupby(["method", "category"])["auroc"].mean().unstack().round(3)
               .reindex(["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]))
    per_cat.to_csv(out_dir / "p5_localizer_by_category.csv")

    best_4nfe = summary.loc[["L1", "L2", "L3", "L4", "L5"], "mean"].idxmax()
    best_4nfe_median = summary.loc[["L1", "L2", "L3", "L4", "L5"], "median"].idxmax()
    best_4nfe_mean = float(summary.loc[best_4nfe, "mean"])
    # The handoff quotes "about 0.79" for L0 and "0.858" for L7; both reproduce to 3 decimals as
    # the *median* over images (mean would be 0.736 / 0.773), so the gate is read on the median.
    best_4nfe_median_value = float(summary.loc[best_4nfe_median, "median"])
    gate = best_4nfe_median_value >= 0.85
    verdict = {"created": now(), "n_images": int(frame["image_id"].nunique()),
               "summary": summary.to_dict(orient="index"),
               "autocorrelation": ac_summary,
               "gate_statistic": "median over images (the handoff's L0~0.79 and L7=0.858 are "
                                 "medians; the corresponding means are 0.736 and 0.773)",
               "best_within_L1_L5_mean_stat": best_4nfe,
               "best_within_L1_L5_mean_auroc": best_4nfe_mean,
               "best_within_L1_L5_median_stat": best_4nfe_median,
               "best_within_L1_L5_median_auroc": best_4nfe_median_value,
               "L6_median": float(summary.loc["L6", "median"]),
               "L7_median": float(summary.loc["L7", "median"]),
               "gate_085_passed_on_median": bool(gate),
               "gate_085_passed_on_mean": bool(best_4nfe_mean >= 0.85),
               "rule": "§2: if any of L1-L6 reaches AUROC >= 0.85 use it as the 4-NFE mask; "
                       "otherwise run both 4-NFE (best of L1-L6) and 10-NFE (L7). The two "
                       "statistics disagree, so §3 runs BOTH arms regardless (costs nothing: "
                       "every arm reuses the P1 residuals)."}
    write_json(out_dir / "p5_localizer_verdict.json", verdict)
    print(f"\nbest single-seed smoothed: {best_4nfe_median} median {best_4nfe_median_value:.4f}  "
          f"(mean {best_4nfe_mean:.4f})  gate 0.85 -> "
          f"{'PASS on median' if gate else 'FAIL'} / "
          f"{'PASS' if best_4nfe_mean >= 0.85 else 'FAIL'} on mean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
