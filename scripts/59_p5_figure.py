"""P5 figure: judge score vs background PSNR, one panel per judge.

Reads ``results/p5/frontier/p5_frontier_curves_<judge>.csv`` (written by
``scripts/55_p5_frontier.py``) and plots the five variants over their lambda grids.

The axis that matters is the *background* PSNR: it is the quantity the variants' amplitude trades
against, so "same background PSNR, higher score" is the only fair comparison between variants.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.io_utils import load_yaml  # noqa: E402

STYLE = {
    "V0": ("#1f77b4", "o", "V0  2 NFE  lam*R0"),
    "V1": ("#2ca02c", "s", "V1  8 NFE  4-seed mean"),
    "S3_9": ("#9467bd", "^", "S3-9  8 NFE  min off-mask seed"),
    "V12": ("#ff7f0e", "D", "V12  10 NFE  mask(t=0.5)*R0"),
    "V2": ("#d62728", "*", "V2  2 NFE  GT mask*R0 (oracle)"),
}
PANELS = [("q_follow", "Qwen2.5-VL: logP(Yes) - logP(No)\n(instruction following)"),
          ("er_dim1", "EditReward head0: instruction following")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out", default="results/p5/fig_frontier.png")
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    src = ROOT / base["output_root"] / "p5" / "frontier"

    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 5.0), dpi=200)
    for ax, (judge, title) in zip(axes, PANELS):
        frame = pd.read_csv(src / f"p5_frontier_curves_{judge}.csv")
        for name, (color, marker, label) in STYLE.items():
            block = frame[frame.variant == name].sort_values("psnr")
            if block.empty:
                continue
            ax.plot(block["psnr"], block["score"], marker=marker, color=color, label=label,
                    lw=1.8, ms=7)
        ax.set_xlabel("background PSNR (dB, higher = better preservation)")
        ax.set_ylabel("judge score (higher = better instruction following)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("P5  §1.4  score vs preservation frontier (200 images, lambda grid, "
                 "no new UNet forward)", fontsize=11)
    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
