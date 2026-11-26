"""P4-E1d analysis: EditReward scores at matched amplitude, paired per image.

All variants are scaled to the same edit norm, so this compares **direction only** — which is
what CLIP-D failed to do.  Judgement is descriptive (the P4 go/no-go was already decided by
step 0), but it answers the question CLIP-D could not: does averaging lose semantics?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402
from geoedit.stats import paired_boot_ci  # noqa: E402

ORDER = ["src", "V0", "V1", "S3_5", "V12", "V2"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / "configs/p3.yaml")
    excluded = set(str(c) for c in p3["exclude_categories"])
    n_boot = int(args.bootstrap)

    out = ROOT / base["output_root"] / "p4" / "e1d"
    frame = pd.concat([pd.read_csv(p, dtype={"image_id": str, "category": str}, on_bad_lines="skip")
                       for p in sorted((out / "shards").glob("shard*.csv"))], ignore_index=True)
    frame = frame.drop_duplicates(subset=["image_id", "variant"])
    frame = frame[~frame["category"].astype(str).isin(excluded)]
    print(f"[e1d] {len(frame)} rows, {frame['image_id'].nunique()} images, "
          f"variants={sorted(frame['variant'].unique())}")

    order = [v for v in ORDER if v in set(frame["variant"])]
    rows = []
    for v in order:
        block = frame[frame.variant == v]
        for dim, col in ((1, "score_dim1"), (2, "score_dim2")):
            ci = paired_boot_ci(block[col].values, n_boot=n_boot, rng=np.random.default_rng(0))
            rows.append({"variant": v, "dim": dim, "mean": ci["point"], "lo": ci["lo"],
                         "hi": ci["hi"], "n": ci["n"],
                         "psnr": float(block["psnr"].mean()),
                         "delta_norm": float(block["delta_norm"].mean())})
    table = pd.DataFrame(rows)
    table.to_csv(out / "p4_e1d_table.csv", index=False)
    print(table.round(3).to_string(index=False))

    # paired differences against the two references people care about
    wide = frame.pivot(index="image_id", columns="variant", values=["score_dim1", "score_dim2"])
    report = {"created": now(), "n_images": int(frame["image_id"].nunique()), "n_boot": n_boot,
              "means": table.groupby(["variant", "dim"])["mean"].first().unstack().to_dict(),
              "paired": {}}
    for dim, col in ((1, "score_dim1"), (2, "score_dim2")):
        for a, b in (("V1", "V0"), ("S3_5", "V0"), ("S3_5", "V1"), ("V2", "V0"),
                     ("V12", "V0"), ("V1", "src"), ("S3_5", "src")):
            if (col, a) not in wide or (col, b) not in wide:
                continue
            diff = (wide[(col, a)] - wide[(col, b)]).dropna()
            ci = paired_boot_ci(diff.values, n_boot=n_boot, rng=np.random.default_rng(1))
            report["paired"].setdefault(f"dim{dim}", {})[f"{a}_minus_{b}"] = {
                "mean": ci["point"], "lo": ci["lo"], "hi": ci["hi"], "n": ci["n"],
                "frac_positive": float((diff > 0).mean())}
    write_json(out / "p4_e1d.json", report)
    print("\npaired (dim1 = instruction following, higher is better):")
    for k, v in report["paired"]["dim1"].items():
        print(f"  {k:14s} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]  "
              f"frac>0={v['frac_positive']:.2f}  n={v['n']}")
    print("\n[reference] P3d CLIP-D said S3-5 - V1 = +0.0156 at the same lambda (amplitude-mismatched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
