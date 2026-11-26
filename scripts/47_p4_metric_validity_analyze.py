"""P4 §5 analysis: CLIP-D per unit amplitude for the three controls vs V0.

Judgement (pre-registered): CLIP-D is a valid semantic metric iff
  (a) E1a (wrong prompt) and E1b (pure noise) both have a per-unit-amplitude CLIP-D
      below 25% of V0's, and
  (b) E1c (reversed) is negative.
Otherwise CLIP-D is an amplitude meter and the semantic matching of step 1 must be redone
with the VLM judge (E1d).
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / "configs/p3.yaml")
    excluded = set(str(c) for c in p3["exclude_categories"])
    n_boot = int(args.bootstrap)

    out = ROOT / base["output_root"] / "p4" / "metric_validity"
    frame = pd.concat([pd.read_csv(p, dtype={"image_id": str, "category": str}, on_bad_lines="skip")
                       for p in sorted((out / "shards").glob("shard*.csv"))], ignore_index=True)
    frame = frame.drop_duplicates(subset=["image_id", "variant", "lam"])
    frame = frame[~frame["category"].astype(str).isin(excluded)]
    frame = frame[frame["variant"] != "recon"]
    print(f"[e1 analyse] {len(frame)} rows, {frame['image_id'].nunique()} images")

    # per-unit-amplitude CLIP-D: ratio of means (the doc's definition) and per-image ratios
    rows = []
    for (variant, lam), block in frame.groupby(["variant", "lam"]):
        per_unit = float(block["clip_d"].mean() / block["delta_norm"].mean())
        rows.append({"variant": variant, "lam": lam, "n": len(block),
                     "clip_d_mean": float(block["clip_d"].mean()),
                     "delta_norm_mean": float(block["delta_norm"].mean()),
                     "clip_d_per_unit": per_unit,
                     "psnr_mean": float(block["psnr"].mean())})
    table = pd.DataFrame(rows).sort_values(["lam", "variant"])
    table.to_csv(out / "p4_metric_validity.csv", index=False)
    print(table.round(4).to_string(index=False))

    # same-lambda ratio against V0, with an image-level bootstrap
    report = {"created": now(), "n_images": int(frame["image_id"].nunique()), "n_boot": n_boot,
              "per_unit": {}, "ratio_vs_V0": {}}
    for lam, block in frame.groupby("lam"):
        v0 = block[block.variant == "V0"].set_index("image_id")
        report["per_unit"][f"{lam:g}"] = {
            v: float(b["clip_d"].mean() / b["delta_norm"].mean())
            for v, b in block.groupby("variant")}
        for variant in ("E1a", "E1b", "E1c"):
            other = block[block.variant == variant].set_index("image_id")
            common = v0.index.intersection(other.index)
            if len(common) < 10:
                continue
            ratio = (other.loc[common, "clip_d"] / other.loc[common, "delta_norm"]) / \
                    (v0.loc[common, "clip_d"] / v0.loc[common, "delta_norm"])
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            ci = paired_boot_ci(ratio.values, n_boot=n_boot, rng=np.random.default_rng(0))
            report["ratio_vs_V0"].setdefault(variant, {})[f"{lam:g}"] = {
                "median": float(np.median(ratio.values)), "mean": ci["point"],
                "lo": ci["lo"], "hi": ci["hi"], "n": ci["n"]}

    lam_main = "1"
    r = report["ratio_vs_V0"]
    a_ok = (r.get("E1a", {}).get(lam_main, {}).get("mean", 9) < 0.25
            and r.get("E1b", {}).get(lam_main, {}).get("mean", 9) < 0.25)
    b_ok = report["per_unit"][lam_main].get("E1c", 9) < 0
    report["verdict"] = {
        "rule": "E1a and E1b per-unit CLIP-D < 25% of V0, and E1c < 0 => CLIP-D is a valid metric",
        "E1a_ratio": r.get("E1a", {}).get(lam_main, {}).get("mean"),
        "E1b_ratio": r.get("E1b", {}).get(lam_main, {}).get("mean"),
        "E1c_per_unit": report["per_unit"][lam_main].get("E1c"),
        "status": "VALID" if (a_ok and b_ok) else "AMPLITUDE_METER",
    }
    write_json(out / "p4_metric_validity.json", report)
    print(f"\n[e1] verdict: {report['verdict']['status']} "
          f"(E1a {report['verdict']['E1a_ratio']}, E1b {report['verdict']['E1b_ratio']}, "
          f"E1c {report['verdict']['E1c_per_unit']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
