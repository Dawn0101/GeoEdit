"""Analysis of the post-hoc experiments (V12/V2d/CLIP-D and the best-of-4 selection).

Everything here is **post-hoc**; it is reported in its own section and never feeds the
pre-registered judgements.  Produces ``results/p3/posthoc/hypotheses.json``.
"""

from __future__ import annotations

import argparse
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


def load_shards(pattern: str) -> pd.DataFrame:
    files = sorted((ROOT / "results/p3/posthoc/shards").glob(pattern))
    frames = [pd.read_csv(p, dtype={"image_id": str, "category": str}, on_bad_lines="skip")
              for p in files if p.stat().st_size > 0]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def curves_of(frame: pd.DataFrame, value: str = "clip_target_edit") -> Dict[str, Dict[str, np.ndarray]]:
    ids = sorted(frame["image_id"].unique())
    lams = sorted(float(v) for v in frame["lam"].unique())
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for variant, block in frame.groupby("variant"):
        piv_c = block.pivot_table(index="lam", columns="image_id", values=value, aggfunc="first")
        piv_p = block.pivot_table(index="lam", columns="image_id", values="psnr", aggfunc="first")
        out[variant] = {"clip": piv_c.reindex(index=lams, columns=ids).to_numpy(dtype=float),
                        "y": piv_p.reindex(index=lams, columns=ids).to_numpy(dtype=float)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    out_dir = ROOT / base["output_root"] / "p3" / "posthoc"
    report: Dict = {"created": now(), "n_boot": args.bootstrap}

    p3 = load_yaml(ROOT / "configs/p3.yaml")
    excluded = set(str(c) for c in p3["exclude_categories"])
    grid = pd.concat([load_shards("grid_shard*.csv"), load_shards("v12c_shard*.csv")],
                     ignore_index=True)
    if not grid.empty:
        grid = grid.drop_duplicates(subset=["image_id", "variant", "lam"])
        grid_main = grid[~grid["category"].astype(str).isin(excluded)]
        report["n_images_all_categories"] = int(grid["image_id"].nunique())
        grid = grid_main
        curves = curves_of(grid)
        report["grid"] = {"n_images": int(grid["image_id"].nunique()), "variants": {}}
        for name, c in curves.items():
            clip_mean = np.nanmean(c["clip"], axis=1)
            psnr_mean = np.nanmean(c["y"], axis=1)
            k = int(np.nanargmax(clip_mean))
            lam = sorted(float(v) for v in grid["lam"].unique())[k]
            report["grid"]["variants"][name] = {
                "max_clip_mean": float(clip_mean[k]),
                "psnr_at_max_clip": float(psnr_mean[k]),
                "lambda_at_max_clip": lam,
                "clip_range": [float(np.nanmin(clip_mean)), float(np.nanmax(clip_mean))],
            }
        names = [n for n in ("V0", "V2", "V2d", "V4", "V12a", "V12b", "V12c") if n in curves]
        report["grid"]["matched_psnr"] = stats.boot_clip_at_psnr(
            curves, names, [25.0, 28.0, 31.0], n_boot=args.bootstrap, rng=np.random.default_rng(0))
        row1 = grid[(grid["lam"] == 1.0)]
        report["grid"]["clip_d_at_lambda1"] = {
            str(v): {"clip_d": float(b["clip_d"].mean()),
                     "clip_target_edit": float(b["clip_target_edit"].mean()),
                     "psnr": float(b["psnr"].mean()), "n": int(len(b))}
            for v, b in row1.groupby("variant")}
        # does a selection rule based on CLIP-D change anything?
        report["grid"]["clip_d_corr"] = float(
            np.corrcoef(row1["clip_d"].astype(float), row1["clip_target_edit"].astype(float))[0, 1])

    best = load_shards("bestof_shard*.csv")
    if not best.empty:
        best = best.drop_duplicates(subset=["image_id", "kind", "lam", "rule"])
        best = best[~best["category"].astype(str).isin(excluded)]
        report["bestof"] = {"n_images": int(best["image_id"].nunique()), "table": {}}
        for (kind, lam, rule), block in best.groupby(["kind", "lam", "rule"]):
            entry = {"n": int(len(block)),
                     "psnr": float(block["psnr"].mean()),
                     "clip_target_edit": float(block["clip_target_edit"].mean()),
                     "clip_d": float(block["clip_d"].mean())}
            for metric in ("d_psnr", "d_clip", "d_clipd"):
                ci = stats.paired_boot_ci(block[metric].astype(float).values,
                                          n_boot=args.bootstrap, rng=np.random.default_rng(0))
                entry[metric] = ci
                entry[f"{metric}_median"] = float(np.median(block[metric].astype(float)))
            report["bestof"]["table"][f"{kind}|{lam:g}|{rule}"] = entry

    write_json(out_dir / "hypotheses.json", report)
    print(f"[posthoc analyse] wrote {out_dir / 'hypotheses.json'} ({now()})")
    if "grid" in report:
        for name, v in report["grid"]["variants"].items():
            print(f"  {name:6s} max CLIP {v['max_clip_mean']:.3f} @lambda={v['lambda_at_max_clip']:g} "
                  f"(PSNR {v['psnr_at_max_clip']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
