"""P5-§1.4 / §3: semantic score vs background PSNR frontiers.

Reads the score sweeps written by ``scripts/52_p5_judge_sweep.py`` and answers the question the
multiplicative-variance corollary stands or falls on: **at the same background PSNR, do the
variant directions score the same?**

Curves are ``(psnr_unedit, judge)`` per variant over the lambda grid.  Comparisons are made at
matched background PSNR by interpolating the mean curve (no extrapolation) inside the PSNR bracket
where all compared variants have support, with an image-level paired bootstrap (2000 draws): each
draw resamples images, rebuilds both mean curves and reports the difference at the fixed target,
so the interval is on the *difference*, not on the two marginals.

Usage:
  ``python scripts/55_p5_frontier.py --spec frontier``   (V0 | V1 | S3-9 | V12 | V2)
  ``python scripts/55_p5_frontier.py --spec ours``       (Ours-4 / Ours-10 vs V0)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402

JUDGE_COLS = {"q_follow": "Qwen2.5-VL following", "er_dim1": "EditReward head0",
              "clip_d": "CLIP-D"}
ORDER = {
    "frontier": ["V0", "V1", "S3_9", "V12", "V2"],
    "ours": ["V0", "O4_k0", "O4_k1", "O4_k2", "O10_k0", "O10_k1", "O10_k2"],
}
PAIRS = {
    "frontier": [("V1", "V0"), ("S3_9", "V0"), ("V12", "V0"), ("V2", "V0"),
                 ("V12", "V1"), ("V2", "V1"), ("S3_9", "V1")],
    "ours": [("O4_k0", "V0"), ("O4_k1", "V0"), ("O4_k2", "V0"), ("O10_k0", "V0"),
             ("O4_k0", "V12"), ("O10_k0", "V12"), ("O4_k1", "O4_k0"), ("O4_k2", "O4_k0")],
}


def _interp(x: np.ndarray, y: np.ndarray, point: float):
    """Mean-curve interpolation with no extrapolation."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 or point < x.min() or point > x.max():
        return None
    order = np.argsort(x)
    return float(np.interp(point, x[order], y[order]))


def boot_delta(curves, a: str, b: str, targets: Sequence[float], n_boot: int = 2000,
               alpha: float = 0.05, rng=None):
    """Paired image-level bootstrap of ``score_a - score_b`` at fixed background PSNR."""
    rng = np.random.default_rng(0) if rng is None else rng
    n_img = curves[a]["psnr"].shape[1]
    draws = {float(t): [] for t in targets}
    for _ in range(n_boot):
        idx = rng.integers(0, n_img, n_img)
        for t in targets:
            pa, sa = np.nanmean(curves[a]["psnr"][:, idx], 1), np.nanmean(curves[a]["score"][:, idx], 1)
            pb, sb = np.nanmean(curves[b]["psnr"][:, idx], 1), np.nanmean(curves[b]["score"][:, idx], 1)
            va, vb = _interp(pa, sa, float(t)), _interp(pb, sb, float(t))
            if va is not None and vb is not None:
                draws[float(t)].append(va - vb)
    out = {}
    for t in targets:
        pa = np.nanmean(curves[a]["psnr"], 1)
        sa = np.nanmean(curves[a]["score"], 1)
        pb = np.nanmean(curves[b]["psnr"], 1)
        sb = np.nanmean(curves[b]["score"], 1)
        va, vb = _interp(pa, sa, float(t)), _interp(pb, sb, float(t))
        values = np.asarray(draws[float(t)], float)
        if va is None or vb is None or values.size < 100:
            out[f"{t:g}"] = {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                             "n_boot": int(values.size), "out_of_range": True}
        else:
            out[f"{t:g}"] = {"delta": float(va - vb),
                             "lo": float(np.quantile(values, alpha / 2)),
                             "hi": float(np.quantile(values, 1 - alpha / 2)),
                             "n_boot": int(values.size), "out_of_range": False}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--spec", choices=["frontier", "ours"], default="frontier")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--targets", default=None, help="comma separated background PSNR values")
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    out_dir = ROOT / base["output_root"] / "p5" / args.spec

    frame = pd.concat([pd.read_csv(p, dtype={"image_id": str, "category": str}, on_bad_lines="skip")
                       for p in sorted((out_dir / "shards").glob("shard*.csv"))], ignore_index=True)
    frame = frame.drop_duplicates(subset=["image_id", "variant", "lam"])
    names = [v for v in ORDER[args.spec] if v in set(frame["variant"])]
    print(f"[{args.spec}] {len(frame)} rows, {frame['image_id'].nunique()} images, {names}")

    lams = sorted(frame[(frame.variant != "recon") & (frame.lam > 0)]["lam"].unique())
    common_ids = sorted(set.intersection(*[set(frame[frame.variant == n]["image_id"].unique())
                                          for n in names]))
    rows, verdicts = [], {}
    for judge in JUDGE_COLS:
        curves: Dict[str, Dict[str, np.ndarray]] = {}
        for name in names:
            block = frame[frame.variant == name]
            ids = common_ids
            block = block[block["image_id"].isin(ids)]
            psnr = np.full((len(lams), len(ids)), np.nan)
            score = np.full_like(psnr, np.nan)
            pos = {i: k for k, i in enumerate(ids)}
            for r, lam in enumerate(lams):
                sel = block[block.lam == lam]
                psnr[r, [pos[i] for i in sel["image_id"]]] = sel["psnr_unedit"].values
                score[r, [pos[i] for i in sel["image_id"]]] = sel[judge].values
            curves[name] = {"psnr": psnr, "score": score}
            for r, lam in enumerate(lams):
                rows.append({"judge": judge, "variant": name, "lam": lam,
                             "psnr": float(np.nanmean(psnr[r])),
                             "score": float(np.nanmean(score[r])),
                             "n": int(np.isfinite(score[r]).sum())})
        table = pd.DataFrame([r for r in rows if r["judge"] == judge])
        table.to_csv(out_dir / f"p5_{args.spec}_curves_{judge}.csv", index=False)

        if args.targets:
            targets = [float(t) for t in args.targets.split(",")]
        else:  # a bracket inside the overlap of every compared variant
            los = [np.nanmean(curves[n]["psnr"], 1).min() for n in names]
            his = [np.nanmean(curves[n]["psnr"], 1).max() for n in names]
            lo, hi = max(los), min(his)
            targets = [float(t) for t in np.linspace(lo + 0.15 * (hi - lo),
                                                     lo + 0.85 * (hi - lo), 3)]
        verdicts[judge] = {"targets_psnr": targets, "pairs": {}}
        for a, b in PAIRS[args.spec]:
            if a not in curves or b not in curves:
                continue
            res = boot_delta(curves, a, b, targets, n_boot=int(args.bootstrap))
            verdicts[judge]["pairs"][f"{a}_minus_{b}"] = res
            print(f"  [{judge}] {a} - {b}")
            for t, v in res.items():
                if v["out_of_range"]:
                    print(f"     at PSNR {t:>6s} dB: out of overlap")
                else:
                    print(f"     at PSNR {t:>6s} dB: {v['delta']:+.3f} "
                          f"[{v['lo']:+.3f}, {v['hi']:+.3f}]")

    write_json(out_dir / f"p5_{args.spec}_frontier.json",
               {"created": now(), "spec": args.spec, "variants": names, "lams": lams,
                "judges": verdicts,
                "reading": "delta > 0 means the first variant scores higher at the same background "
                           "PSNR; the interval is a paired image-level bootstrap on the difference."})
    print(f"\nwrote {out_dir}/p5_{args.spec}_frontier.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
