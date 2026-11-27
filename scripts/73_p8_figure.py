"""P8 figure: the three bug fixes and where the remaining gap is.

(a) the λ=1 same-table position (background PSNR vs CLIP-Edited) for every corrected variant, the
    buggy P7 point and ChordEdit -- the whole point of P8 is that this moved;
(b) the paired A/B against ChordEdit on the two axes the PI asked for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.io_utils import load_yaml  # noqa: E402

LABELS = {
    "b_k1_lam1_vs_chord@22": "b_k1 @lam=1  vs  ChordEdit",
    "b_k1_dino_vs_chord@22": "b_k1 @DINO-matched  vs  ChordEdit",
    "b_k0_dino_vs_chord@22": "b_k0 @DINO-matched  vs  ChordEdit",
    "b_k1_lam1_vs_b_k0_lam1": "b_k1  vs  b_k0  (SDS on/off)",
    "b_k1_lam1_vs_b_k1_fullprox_lam1": "masked prox  vs  full prox",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out", default="results/p8/fig_p8.png")
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    p8 = ROOT / base["output_root"] / "p8b"

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=200)
    ax = axes[0]
    table = pd.read_csv(p8 / "ab_table.csv") if (p8 / "ab_table.csv").exists() else None
    summ = pd.read_csv(p8 / "frontier" / "p8b_frontier_summary.csv") if (
        p8 / "frontier" / "p8b_frontier_summary.csv").exists() else None
    if summ is None:
        import glob
        f = pd.concat([pd.read_csv(x, dtype={"image_id": str})
                       for x in sorted((p8 / "frontier").glob("per_image*.csv"))])
        summ = (f.drop_duplicates(subset=["image_id", "run", "lam"])
                .groupby(["run", "lam"])[["psnr_unedit", "clip_target_edit", "struct_dist"]]
                .mean().reset_index())
    at1 = summ[np.isclose(summ["lam"], 1.0)]
    style = {"b_k0": ("#1f77b4", "o"), "b_k05": ("#2ca02c", "s"), "b_k1": ("#d62728", "^"),
             "b_k1_fullprox": ("#9467bd", "D"), "b_k1_noprox": ("#8c564b", "v")}
    for _, r in at1.iterrows():
        c, m = style.get(r["run"], ("gray", "x"))
        ax.scatter(r["psnr_unedit"], r["clip_target_edit"], s=170, color=c, marker=m,
                   edgecolor="k", zorder=4, label=r["run"])
    chord_psnr, chord_clip = 22.02, 22.59
    ax.scatter([chord_psnr], [chord_clip], marker="*", s=520, color="gold", edgecolor="k",
               zorder=5, label="ChordEdit")
    ax.scatter([26.75], [20.13], marker="X", s=200, color="black", zorder=5,
               label="P7 o5 @lam=1 (buggy)")
    ax.annotate("P7 bug:\nbelow unedited\n(src 20.26)", (26.75, 20.13), textcoords="offset points",
                xytext=(8, 18), fontsize=8)
    ax.set_xlabel("background PSNR (dB, higher = better preservation)")
    ax.set_ylabel("CLIP-Edited (higher = stronger semantics)")
    ax.set_title("(a) lambda=1 same table: the fixes moved us up and left", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    if table is not None:
        t = table.sort_values("s")
        y = np.arange(len(t))
        ax.errorbar(t["s"], y, xerr=[t["s"] - t["lo"], t["hi"] - t["s"]], fmt="o",
                    color="k", ecolor="k", capsize=4, ms=7, zorder=3)
        ax.scatter(t["s"], y, c=["#2ca02c" if v > 0 else "#d62728" for v in t["s"]], s=90, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS.get(p, p) for p in t["pair"]], fontsize=9)
        for yi, (v, w) in enumerate(zip(t["s"], t["win"])):
            ax.annotate(f"{v:+.2f} (win {w:.0%})", (v, yi), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)
    ax.axvline(0.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("A/B score  s  (positive = Ours better)")
    ax.set_title("(b) VLM paired comparison, 200 images", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("P8  after fixing the three P7 bugs (chord field, quantile mask, convex mix)", fontsize=12)
    fig.tight_layout()
    out = ROOT / args.out
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
