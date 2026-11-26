"""P4 step 0: is there a usable closed-loop signal at all?  (offline, no new forwards)

Handoff §3.  Reuses the fp32 residuals cached by ``30_p3a_build_and_eval.py --prep``
(4 seeds x t in {0.5, 0.7, 0.9}), 540 images of categories 0-7.

Statistics, all compared against the GT latent mask (continuous AUROC):

``P1pair/P3pair/P6pair``  ``pool3(<R_i, R_j>)`` averaged over 1 / 3 / 6 seed pairs (i != j).
                          One pair is roughly the evidence a single closed-loop round has.
``X``                     cross-scale ``pool3(<R(0.9)_i, R(0.5)_j>)`` over all 4x4 pairs.
``Ratio``                 ``sqrt(E||R(0.5)||^2 / E||R(0.9)||^2)`` per pixel, then pool3.
``E``                     DiffEdit-style mean ``||R||`` (the P3d reference: 0.858/0.855/0.781).

Gate (§3.2): ``P6pair`` at t=0.9 >= 0.8 -> go; < 0.7 at all three t -> stop.
Also calibrates ``alpha`` on a separate 50-image set (§3.3, GT is used there **once**).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import masks  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-pixel inner product summed over the 4 channels -> [64,64]."""
    return (a * b).sum(dim=0)


def pool3(x: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    return F.avg_pool2d(x[None, None], kernel_size=3, stride=1, padding=1,
                        count_include_pad=False)[0, 0]


def stats_for_image(bundle, seeds=(0, 1, 2, 3)) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    R = {t: torch.stack([bundle[f"t{t:g}_s{s}"] for s in seeds], 0).float()
         for t in (0.5, 0.7, 0.9)}
    for t, Rs in R.items():
        pooled = [pool3(inner(Rs[i], Rs[j])) for i, j in PAIRS]
        for name, k in (("P1pair", 1), ("P3pair", 3), ("P6pair", 6)):
            out[f"{name}_t{t:g}"] = torch.stack(pooled[:k]).mean(0)
        out[f"E_t{t:g}"] = Rs.norm(dim=1).mean(0)
    cross = [pool3(inner(R[0.9][i], R[0.5][j])) for i in range(4) for j in range(4)]
    out["X"] = torch.stack(cross).mean(0)
    e9 = (R[0.9] ** 2).sum(dim=(0, 1)).mean() if False else (R[0.9] ** 2).sum(dim=1).mean(0)
    e5 = (R[0.5] ** 2).sum(dim=1).mean(0)
    out["Ratio"] = pool3((e5.clamp_min(0) / e9.clamp_min(1e-12)).sqrt())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p3-config", default="configs/p3.yaml")
    parser.add_argument("--n-calib", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=200)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / "configs/p3.yaml")
    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    a_root = ROOT / base["output_root"] / "p3" / "a"

    pool = [i for i in range(len(dataset)) if dataset[i].category not in ("8", "9")]
    # the eval set: the same 200 images as criterion (1) in P3d (deterministic linspace)
    eval_idx = [pool[i] for i in np.linspace(0, len(pool) - 1, args.n_eval).round().astype(int).tolist()]
    eval_ids = {dataset[i].image_id for i in eval_idx}
    rest = [i for i in pool if dataset[i].image_id not in eval_ids]
    # the alpha calibration set: 50 images, fixed seed, disjoint from the eval set
    rng = np.random.default_rng(20261125)
    calib_idx = [rest[i] for i in rng.choice(len(rest), args.n_calib, replace=False)]
    calib_ids = [dataset[i].image_id for i in calib_idx]

    keys = ["P1pair_t0.9", "P3pair_t0.9", "P6pair_t0.9", "E_t0.9",
            "P1pair_t0.7", "P3pair_t0.7", "P6pair_t0.7", "E_t0.7",
            "P1pair_t0.5", "P3pair_t0.5", "P6pair_t0.5", "E_t0.5",
            "X", "Ratio"]
    per_image: Dict[str, List[float]] = {k: [] for k in keys}
    alpha_scores: List[Dict[str, float]] = []

    for index in pool:
        sample = dataset[index]
        path = a_root / "R" / f"{sample.image_id}.pt"
        if not path.exists():
            continue
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        s = stats_for_image(bundle)
        gt = sample.mask64
        for k in keys:
            per_image[k].append(masks.auroc(masks.to_numpy(s[k]), gt))

        if sample.image_id in calib_ids:
            # one pair, t=0.9, normalised by the median magnitude (as in the handover)
            raw = s["P1pair_t0.9"]
            normalised = raw / (raw.abs().median().clamp_min(1e-8))
            m = torch.from_numpy(gt).float()
            inside = float(normalised[m > 0.5].mean()) if bool((m > 0.5).any()) else float("nan")
            outside = float(normalised[m <= 0.5].mean()) if bool((m <= 0.5).any()) else float("nan")
            alpha_scores.append({"image_id": sample.image_id, "mu_in": inside, "mu_out": outside})

    rows = []
    for k in keys:
        values = np.asarray(per_image[k], dtype=float)
        values = values[np.isfinite(values)]
        rows.append({"statistic": k, "n": int(values.size), "auroc_median": float(np.median(values)),
                     "auroc_mean": float(values.mean()),
                     "auroc_p25": float(np.quantile(values, 0.25)),
                     "auroc_p75": float(np.quantile(values, 0.75))})
    import pandas as pd

    table = pd.DataFrame(rows)
    out = ROOT / base["output_root"] / "p4" / "step0"
    table.to_csv(out / "p4_step0_auroc.csv", index=False)
    print(table.to_string(index=False))

    inside = np.array([r["mu_in"] for r in alpha_scores], dtype=float)
    outside = np.array([r["mu_out"] for r in alpha_scores], dtype=float)
    alpha = float((np.nanmean(inside) + np.nanmean(outside)) / 2)
    calib = {"created": now(), "n_calib": len(alpha_scores), "calib_ids": calib_ids,
             "mu_in": float(np.nanmean(inside)), "mu_out": float(np.nanmean(outside)),
             "alpha": alpha,
             "note": "alpha uses GT on a separate 50-image set, once (handoff §3.3)"}
    write_json(out / "p4_alpha_calib.json", calib)

    p6 = {k: float(table[table.statistic == f"P6pair_t{k}"]["auroc_median"].iloc[0]) for k in ("0.9", "0.7", "0.5")}
    e = {k: float(table[table.statistic == f"E_t{k}"]["auroc_median"].iloc[0]) for k in ("0.9", "0.7", "0.5")}
    verdict = ("GO" if p6["0.9"] >= 0.8 else ("STOP" if all(v < 0.7 for v in p6.values()) else "GO_WITH_H3_PRIMARY"))
    write_json(out / "p4_step0_verdict.json", {
        "created": now(), "P6pair": p6, "E": e, "gate_rule": "P6pair(t=0.9) >= 0.8 -> GO; all three < 0.7 -> STOP",
        "verdict": verdict, "alpha": alpha})
    print(f"\n[step0] P6pair: " + ", ".join(f"t={k}: {v:.3f}" for k, v in p6.items()))
    print(f"[step0] E(ref) : " + ", ".join(f"t={k}: {v:.3f}" for k, v in e.items()))
    print(f"[step0] alpha = {alpha:.4f} (mu_in {calib['mu_in']:.3f}, mu_out {calib['mu_out']:.3f})")
    print(f"[step0] VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
