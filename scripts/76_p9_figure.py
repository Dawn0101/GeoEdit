"""P9 figure: (a) the official-pipeline variants on the fidelity/semantics plane, (b) the VLM pairs."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from geoedit.io_utils import load_yaml

LAB = {"m075_k0_vs_off_k0": "Chord+Ours-Mask (k=0)  vs  Official-Chord",
       "m075_k10_vs_off_k0": "Chord+Ours-Mask+SDS (k=1)  vs  Official-Chord",
       "m05_k0_vs_m075_k0": "mask from t=0.5  vs  mask from t=0.75",
       "m075_k0_vs_m075_k10": "SDS k=1  vs  k=0  (both masked)"}
STYLE = {"off_k0": ("gold", "*", 520), "m075_k0": ("#1f77b4", "o", 170),
         "m075_k05": ("#2ca02c", "s", 150), "m075_k10": ("#d62728", "^", 150),
         "m05_k0": ("#9467bd", "D", 150)}

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="results/p9/fig_p9.png"); a = ap.parse_args()
    base = load_yaml(ROOT / a.config); p9 = ROOT / base["output_root"] / "p9"
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=200)
    ax = axes[0]
    s = pd.read_csv(p9 / "frontier" / "summary.csv")
    for _, r in s.iterrows():
        c, m, sz = STYLE.get(r["variant"], ("gray", "x", 120))
        ax.scatter(r["psnr_unedit"], r["clip_target_edit"], s=sz, color=c, marker=m,
                   edgecolor="k", zorder=4, label=r["variant"])
    ax.set_xlabel("background PSNR (dB, higher = better preservation)")
    ax.set_ylabel("CLIP-Edited (higher = stronger semantics)")
    ax.set_title("(a) on the OFFICIAL ChordEdit pipeline (200 images)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    ax = axes[1]
    t = None
    for cand in sorted(p9.glob("ab_table*.csv")) + sorted((ROOT / base["output_root"]).glob("p9/ab_table*.csv")):
        t = pd.read_csv(cand); break
    if t is not None:
        t = t[t.subset == "all"].sort_values("s")
        y = np.arange(len(t))
        ax.errorbar(t["mean_s"], y, xerr=[t["mean_s"] - t["lo"], t["hi"] - t["mean_s"]],
                    fmt="o", color="k", ecolor="k", capsize=4, ms=7, zorder=3)
        ax.scatter(t["mean_s"], y, c=["#2ca02c" if v > 0 else "#d62728" for v in t["mean_s"]],
                   s=90, zorder=4)
        ax.set_yticks(y); ax.set_yticklabels([LAB.get(p, p) for p in t["pair"]], fontsize=8.5)
        for yi, (v, w) in enumerate(zip(t["mean_s"], t["win_rate"])):
            ax.annotate(f"{v:+.2f} (win {w:.0%})", (v, yi), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("A/B score  s  (positive = first variant better)")
    ax.set_title("(b) VLM paired preference, 200 images", fontsize=10); ax.grid(alpha=0.3, axis="x")
    fig.suptitle("P9  the 1.8-point gap traced to the text-encoding mask; mask on the official pipeline", fontsize=11.5)
    fig.tight_layout(); out = ROOT / a.out; fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
