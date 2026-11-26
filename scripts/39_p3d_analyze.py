"""P3d analysis: D1a / D1b / D1c (§2 of the handoff) with image-level bootstrap.

Everything is judged on **CLIP-D** (the sensitive semantic metric); CLIP-Edited is reported
next to it for comparison.  ``lam``-matched and frontier-matched comparisons are kept apart:
``D1a`` is a same-lambda paired comparison, ``D1b`` is a frontier comparison at a fixed
CLIP-D anchor (no extrapolation; ``out_of_range`` when a curve does not cover it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import stats  # noqa: E402
from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402

VARIANTS = ["V0", "V1", "V1x", "V12", "S3_5", "S3_9", "H1_5", "H1_9", "H0_5", "V2"]
NFE = {"V0": 2, "V1": 8, "V1x": 16, "V12": 10, "S3_5": 16, "S3_9": 8,
       "H1_5": 16, "H1_9": 8, "H0_5": 16}


def matrices(frame: pd.DataFrame, value: str, variants: List[str]):
    ids = sorted(frame["image_id"].unique())
    lams = sorted(float(v) for v in frame["lam"].unique())
    out = {}
    for name in variants:
        block = frame[frame["variant"] == name]
        if block.empty:
            continue
        piv = block.pivot_table(index="lam", columns="image_id", values=value, aggfunc="first")
        out[name] = piv.reindex(index=lams, columns=ids).to_numpy(dtype=float)
    return lams, ids, out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / "configs/p3.yaml")
    excluded = set(str(c) for c in p3["exclude_categories"])
    n_boot = int(args.bootstrap)

    files = sorted((ROOT / base["output_root"] / "p3d" / "d1" / "shards").glob("shard*.csv"))
    frame = pd.concat([pd.read_csv(p, dtype={"image_id": str, "category": str}, on_bad_lines="skip")
                       for p in files if p.stat().st_size > 0], ignore_index=True)
    frame = frame.drop_duplicates(subset=["image_id", "variant", "lam"])
    frame = frame[~frame["category"].astype(str).isin(excluded)]
    variants = [v for v in VARIANTS if v in set(frame["variant"])]
    print(f"[d1 analyse] {len(frame)} rows, {frame['image_id'].nunique()} images, variants={variants}")

    lams, ids, clipd = matrices(frame, "clip_d", variants)
    _, _, psnr = matrices(frame, "psnr", variants)
    _, _, clipw = matrices(frame, "clip_target_edit", variants)
    lam1 = lams.index(1.0) if 1.0 in lams else None

    report: Dict = {"created": now(), "n_images": int(len(ids)), "n_boot": n_boot,
                    "lambda": lams, "nfe": NFE, "variants": {}}

    # ---- table: mean +- CI per (variant, lambda) ----
    for name in variants:
        rows = []
        for k, lam in enumerate(lams):
            values = clipd[name][k]
            values = values[np.isfinite(values)]
            ci = stats.boot_ci(values, n_boot=n_boot, statistic=np.mean,
                               rng=np.random.default_rng(0))
            rows.append({"lam": lam, "clip_d": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
                         "psnr": float(np.nanmean(psnr[name][k])),
                         "clip_edit": float(np.nanmean(clipw[name][k])),
                         "delta_norm_over_lam": float(
                             frame[(frame.variant == name) & (frame.lam == lam)]["delta_norm_over_lam"].mean())})
        report["variants"][name] = {"nfe": NFE.get(name), "curve": rows}

    # ---- D1a: same lambda, CLIP-D(S3_5) - CLIP-D(V1) > 0 ----
    if "S3_5" in variants and "V1" in variants and lam1 is not None:
        diff = clipd["S3_5"][lam1] - clipd["V1"][lam1]
        ci = stats.paired_boot_ci(diff, n_boot=n_boot, rng=np.random.default_rng(1))
        report["D1a"] = {"rule": "CLIP-D(S3_5) - CLIP-D(V1) > 0 at lambda=1",
                         "status": ("SUPPORTED" if ci["lo"] > 0 else
                                    ("REFUTED" if ci["hi"] < 0 else "INCONCLUSIVE")),
                         "detail": ci}
        # the same at every lambda, as a sanity check on monotonicity
        report["D1a"]["by_lambda"] = {
            f"{lam:g}": stats.paired_boot_ci(clipd["S3_5"][k] - clipd["V1"][k], n_boot=n_boot,
                                             rng=np.random.default_rng(2))
            for k, lam in enumerate(lams)}

    # ---- D1b: frontier comparison at V0's lambda=1 CLIP-D anchor ----
    if lam1 is not None:
        curves = {n: {"clip": clipd[n], "y": psnr[n]} for n in variants}
        match = stats.mean_curve_match(curves, "V0", anchor_index=[lam1], n_boot=n_boot,
                                       rng=np.random.default_rng(3))

        def paired(a: str, b: str) -> Dict[str, float]:
            ad = np.asarray(match["boot"][str(lam1)].get(a) or [], dtype=float)
            bd = np.asarray(match["boot"][str(lam1)].get(b) or [], dtype=float)
            length = min(ad.size, bd.size)
            if length == 0:
                return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0,
                        "out_of_range": True}
            diff = ad[:length] - bd[:length]
            return {"point": float(diff.mean()), "lo": float(np.quantile(diff, 0.025)),
                    "hi": float(np.quantile(diff, 0.975)), "n": int(length), "out_of_range": False}

        anchor = float(match["anchor_clip"][str(lam1)])
        cmp = {}
        for a, b in (("H1_5", "V1x"), ("H1_9", "V1"), ("S3_5", "V1"), ("S3_9", "V1"),
                     ("V12", "V1"), ("H0_5", "V1")):
            if a in variants and b in variants:
                cmp[f"{a}_vs_{b}"] = paired(a, b)
        best = max((v for k, v in cmp.items() if np.isfinite(v["point"])),
                   key=lambda v: v["point"], default=None)
        report["D1b"] = {
            "rule": "dPSNR(H1 - V1x) >= +1 dB at the CLIP-D anchor (V0 at lambda=1), CI > 0",
            "anchor_clip_d": anchor,
            "at_anchor": {n: {"psnr": match["results"][str(lam1)][n].psnr_at_anchor,
                              "out_of_range": match["results"][str(lam1)][n].out_of_range}
                          for n in variants if str(lam1) in match["results"]},
            "comparisons": cmp,
            "status": ("SUPPORTED" if cmp.get("H1_5_vs_V1x", {}).get("lo", -1) >= 1.0 else
                       "REFUTED" if cmp.get("H1_5_vs_V1x", {}).get("hi", 1) < 1.0 else "INCONCLUSIVE"),
            "best": best,
        }

    # ---- D1c: per category at lambda=1 ----
    if lam1 is not None:
        sub = frame[frame["lam"] == 1.0]
        table = {}
        for name in [v for v in ("V0", "V1", "S3_5", "H1_5", "V2") if v in variants]:
            block = sub[sub["variant"] == name]
            table[name] = {str(cat): {"psnr": float(b["psnr"].mean()),
                                      "clip_d": float(b["clip_d"].mean()),
                                      "n": int(len(b))}
                           for cat, b in block.groupby("category")}
        report["D1c"] = table

    out_dir = ROOT / base["output_root"] / "p3d"
    write_json(out_dir / "d1_hypotheses.json", report)
    # flat table for the package
    rows = []
    for name, block in report["variants"].items():
        for r in block["curve"]:
            rows.append({"variant": name, "nfe": block["nfe"], **r})
    pd.DataFrame(rows).to_csv(out_dir / "p3d_d1_table.csv", index=False)
    print(f"[d1 analyse] D1a={report.get('D1a', {}).get('status')} "
          f"D1b={report.get('D1b', {}).get('status')}")
    print(f"[d1 analyse] wrote {out_dir / 'd1_hypotheses.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
