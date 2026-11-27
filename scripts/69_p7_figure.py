"""P7 figure: Ours-5 vs ChordEdit.

(a) the matched-background-PSNR forest plot (same layout as P6 so the two rounds compare directly);
(b) the semantic-vs-fidelity trade-off: CLIP-Edited against background PSNR for every variant over
    the lambda grid, with ChordEdit as a single fixed point (it has no lambda knob).
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
    "o5@22.0_vs_chord@22": "Ours-5  vs  ChordEdit",
    "o5@22.0_vs_o5_fullprox@22.0": "Ours-5  vs  full-image prox",
    "o5@22.0_vs_o5_noprox@22.0": "Ours-5  vs  no prox (4 NFE)",
    "o5@22.0_vs_o5_nogate@22.0": "Ours-5  vs  no direction gate",
    "o5@22.0_vs_o5_nofuse@22.0": "Ours-5  vs  single-scale",
}
STYLE = {"o5": ("#2ca02c", "o"), "o5_fullprox": ("#9467bd", "s"), "o5_noprox": ("#8c564b", "^"),
         "o5_nogate": ("#17becf", "P"), "o5_nofuse": ("#bcbd22", "X"), "o5_g3": ("#7f7f7f", "v")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out", default="results/p7/fig_p7_ab.png")
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    p7 = ROOT / base["output_root"] / "p7"

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=200)
    ax = axes[0]
    table = pd.read_csv(p7 / "ab" / "p7_ab_table.csv")
    sub = table[table.subset == "all"]
    y = np.arange(len(sub))
    ax.errorbar(sub["mean_s"], y, xerr=[sub["mean_s"] - sub["lo"], sub["hi"] - sub["mean_s"]],
                fmt="o", color="k", ecolor="k", elinewidth=1.4, capsize=4, ms=7, zorder=3)
    ax.scatter(sub["mean_s"], y, c=["#2ca02c" if m > 0 else "#d62728" for m in sub["mean_s"]],
               s=90, zorder=4)
    ax.axvline(0.0, color="gray", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(p, p) for p in sub["pair"]], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("A/B score  s  (positive = first variant better)")
    ax.set_title("(a) matched background PSNR, 200 images, 2000 boots", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    for yi, (m, w) in enumerate(zip(sub["mean_s"], sub["win_rate"])):
        ax.annotate(f"{m:+.2f} (win {w:.0%})", (m, yi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)

    ax = axes[1]
    frames = [pd.read_csv(p, dtype={"image_id": str})
              for p in sorted((p7 / "frontier").glob("per_image*.csv"))]
    frame = pd.concat(frames, ignore_index=True)
    curve = frame.groupby(["run", "lam"])[["psnr_unedit", "clip_target_edit"]].mean().reset_index()
    for run, block in curve.groupby("run"):
        color, marker = STYLE.get(run, ("gray", "x"))
        ax.plot(block["psnr_unedit"], block["clip_target_edit"], marker=marker, color=color,
                lw=1.6, ms=6, label=run, alpha=0.9)
    chord = pd.read_csv(ROOT / "results/p6/frontier/runs_chord.csv")
    ax.scatter([chord["psnr_unedit"].mean()], [chord["clip_target_edit"].mean()], marker="*",
               s=420, color="#d62728", edgecolor="k", zorder=6, label="ChordEdit (5 NFE)")
    ax.set_xlabel("background PSNR (dB, higher = better preservation)")
    ax.set_ylabel("CLIP-Edited (higher = stronger semantics)")
    ax.set_title("(b) the gap is semantics, not fidelity", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle("P7  Ours-5 (5 UNet forwards) vs ChordEdit (5 UNet forwards)", fontsize=12)
    fig.tight_layout()
    out = ROOT / args.out
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
