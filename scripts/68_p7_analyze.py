"""P7 analysis: matched-PSNR lambda solving for Ours-5 and the ChordEdit head-to-head.

``--mode matched``  concatenates the P7 frontier shards, builds ``background PSNR(lam)`` per
                    variant and solves the lambda that hits each target on the **mean** curve.
``--mode report``   turns ``results/p7/ab/*.csv`` into the paired-bootstrap table (same statistics
                    as P6: ``s`` is the paired difference of the validated Yes/No instrument).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402
from geoedit.stats import boot_ci  # noqa: E402
from importlib import import_module  # noqa: E402

_interp = import_module("65_p6_analyze")._interp


def load_frontier(frontier_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(frontier_dir.glob("per_image*.csv")):
        frames.append(pd.read_csv(path, dtype={"image_id": str}, on_bad_lines="skip"))
    if not frames:
        raise SystemExit(f"no per_image*.csv in {frontier_dir}")
    frame = pd.concat(frames, ignore_index=True)
    return frame.drop_duplicates(subset=["image_id", "run", "lam"])


def matched(out: Path, frontier_dir: Path, targets: List[float]) -> int:
    frame = load_frontier(frontier_dir)
    curve = frame.groupby(["run", "lam"])["psnr_unedit"].mean().reset_index()
    pivot = curve.pivot(index="run", columns="lam", values="psnr_unedit")
    pivot.to_csv(out / "p7_curves_psnr.csv")
    print(f"[matched] {len(pivot)} runs x {len(pivot.columns)} lams, "
          f"{frame['image_id'].nunique()} images")
    print(pivot.round(2).to_string())
    rows = []
    for run, series in pivot.iterrows():
        xs = [float(c) for c in pivot.columns]
        ys = [float(series[c]) for c in pivot.columns]
        for t in targets:
            lam = _interp(ys, xs, t)  # psnr -> lam
            if lam is None:
                continue
            rows.append({"run": run, "target": t, "lam": lam})
    spec = pd.DataFrame(rows)
    spec.to_csv(out / "p7_matched_lam.csv", index=False)
    print(f"\n[matched] {len(spec)} (run, target, lam) rows")
    print(spec.pivot(index="run", columns="target", values="lam").round(3).to_string())
    write_json(out / "p7_matched_meta.json", {"created": now(), "targets": targets,
                                              "n_images": int(frame["image_id"].nunique()),
                                              "lams": sorted(float(x) for x in pivot.columns)})
    return 0


def report(out: Path, n_boot: int, tol: float) -> int:
    files = [p for p in sorted(out.glob("*.csv")) if p.stem.startswith(("p7_ab", "pairs"))]
    rows = []
    for path in files:
        frame = pd.read_csv(path, dtype={"image_id": str, "category": str})
        if "pair" not in frame.columns or "image_id" not in frame.columns:
            continue
        frame = frame.drop_duplicates(subset=["image_id", "pair"])
        for pair, block in frame.groupby("pair"):
            for subset, sel in (("all", np.ones(len(block), bool)),
                                (f"|dPSNR|<{tol:g}",
                                 (block["psnr_a"] - block["psnr_b"]).abs() < tol)):
                values = block.loc[sel, "s"].values
                if values.size < 20:
                    continue
                ci = boot_ci(values, n_boot=n_boot, rng=np.random.default_rng(0))
                rows.append({"file": path.stem, "pair": pair, "subset": subset, "n": int(values.size),
                             "mean_s": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
                             "win_rate": float((values > 0).mean()),
                             "mean_dpsnr": float((block.loc[sel, "psnr_a"] -
                                                  block.loc[sel, "psnr_b"]).mean()),
                             "psnr_a": float(block.loc[sel, "psnr_a"].mean()),
                             "psnr_b": float(block.loc[sel, "psnr_b"].mean())})
    if not rows:
        print("[report] no pair files")
        return 1
    table = pd.DataFrame(rows)
    table.to_csv(out / "p7_ab_table.csv", index=False)
    print(table.round(3).to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", choices=["matched", "report"], default="matched")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--tol", type=float, default=1.0)
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    p7 = load_yaml(ROOT / "configs/p7.yaml")
    out = ROOT / base["output_root"] / "p7" / "ab"
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "matched":
        targets = ([float(t) for t in args.targets.split(",")] if args.targets
                   else [float(t) for t in p7["matched_psnr"]["targets"]])
        return matched(out, ROOT / base["output_root"] / "p7" / "frontier", targets)
    return report(out, int(args.bootstrap), float(args.tol))


if __name__ == "__main__":
    sys.exit(main())
