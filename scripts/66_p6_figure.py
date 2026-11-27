"""P6 figure: the matched-background-PSNR A/B comparison.

Two panels:
  (a) the pre-registered comparisons as a forest plot (``s`` with its paired-bootstrap CI);
  (b) the 50-pair k-sweep: background PSNR against CLIP-Edited, which shows the *cost* side
      (k buys semantics but spends preservation, gamma=2 buys most of it back).
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
    "ours4_k1_g1@24.0_vs_ours4_k0_g1@24.0": "k=1  vs  k=0   (SDS on/off)",
    "ours4_k2_g1@24.0_vs_ours4_k1_g1@24.0": "k=2  vs  k=1",
    "ours4_k0_g1@24.0_vs_v0@24.0": "masked k=0  vs  V0",
    "ours4_k1_g1@22.0_vs_chord@22": "Ours-4(k=1)  vs  ChordEdit",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out", default="results/p6/fig_p6_ab.png")
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    p6_dir = ROOT / base["output_root"] / "p6"

    table = pd.read_csv(p6_dir / "ab" / "p6_ab_table.csv")
    sub = table[table.subset == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=200)

    ax = axes[0]
    names = [LABELS.get(p, p) for p in sub["pair"]]
    y = np.arange(len(sub))
    colors = ["#2ca02c" if m > 0 else "#d62728" for m in sub["mean_s"]]
    ax.errorbar(sub["mean_s"], y, xerr=[sub["mean_s"] - sub["lo"], sub["hi"] - sub["mean_s"]],
                fmt="o", color="k", ecolor="k", elinewidth=1.4, capsize=4, ms=7, zorder=3)
    ax.scatter(sub["mean_s"], y, c=colors, s=90, zorder=4)
    ax.axvline(0.0, color="gray", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("A/B score  s  (positive = first variant follows the instruction better)")
    ax.set_title("(a) matched background PSNR, 200 images, 2000 boots", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    for yi, (m, w) in enumerate(zip(sub["mean_s"], sub["win_rate"])):
        ax.annotate(f"{m:+.2f}  (win {w:.0%})", (m, yi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)

    ax = axes[1]
    vis = pd.read_csv(p6_dir / "vis_k" / "p6_vis_k.csv")
    style = {"k=0": ("#1f77b4", "o"), "k=1": ("#2ca02c", "s"), "k=2": ("#d62728", "^"),
             "k=2 g=2": ("#ff7f0e", "D")}
    for variant, block in vis.groupby("variant"):
        c, m = style.get(variant, ("gray", "x"))
        ax.scatter(block["psnr_unedit"], block["clip_target_edit"], s=14, alpha=0.55,
                   color=c, marker=m, label=variant)
        ax.scatter([block["psnr_unedit"].mean()], [block["clip_target_edit"].mean()], s=200,
                   color=c, marker=m, edgecolor="k", zorder=5)
    ax.set_xlabel("background PSNR (dB, higher = better preservation)")
    ax.set_ylabel("CLIP-Edited (higher = stronger semantic change)")
    ax.set_title("(b) 50 pairs at lam=1: the k trade-off", fontsize=10)
    ax.legend(fontsize=8, loc="lower left", title="big marker = mean")
    ax.grid(alpha=0.3)

    fig.suptitle("P6  Ours-4: in-mask SDS at 4 NFE", fontsize=12)
    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
