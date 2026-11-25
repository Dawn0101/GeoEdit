"""Stage E: P2 analysis -- summary table, Pareto figures, and the H2a-H2d verdicts.

Reads ``results/p2/per_image.csv`` (merged from the shards) and the generation diagnostics,
then writes everything §9.2 asks for.  The pre-registered rules of §4.3 are implemented
literally; each block cites its rule.

Bootstrap convention: resample **images** with replacement (the unit of independence),
1000 draws, percentile CI.  For H2b both curves are recomputed inside every draw.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from geoedit.io_utils import load_yaml, read_rows, write_json
from geoedit.stats import boot_ci, delta_vs_curve, paired_boot_ci, wilcoxon_p

ROOT = Path(__file__).resolve().parents[1]
N_BOOT_DEFAULT = 1000


def pareto_delta(
    clip: np.ndarray,
    psnr: np.ndarray,
    lam_clip: np.ndarray,
    lam_psnr: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """ΔPSNR of the point vs the λ curve matched on CLIP, with an image-level bootstrap.

    ``clip``/``psnr``: per-image arrays of the point being tested.
    ``lam_clip``/``lam_psnr``: ``[n_images, n_lambda]`` arrays of the λ curve.
    """
    rng = np.random.default_rng(seed)
    n = len(clip)
    lam_mean_clip = np.nanmean(lam_clip, axis=0)
    lam_mean_psnr = np.nanmean(lam_psnr, axis=0)
    mean_clip = float(np.nanmean(clip))
    mean_psnr = float(np.nanmean(psnr))
    matched = delta_vs_curve(lam_mean_clip, lam_mean_psnr, mean_clip, mean_psnr)
    out_of_range = matched.out_of_range
    point_delta = matched.delta
    point_matched = matched.psnr_curve_at

    boots = []
    if not out_of_range:
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, n)
            curve_clip = np.nanmean(lam_clip[idx], axis=0)
            curve_psnr = np.nanmean(lam_psnr[idx], axis=0)
            b_matched = delta_vs_curve(curve_clip, curve_psnr,
                                       float(np.nanmean(clip[idx])),
                                       float(np.nanmean(psnr[idx])))
            boots.append(b_matched.delta if not b_matched.out_of_range else np.nan)
    boots = np.array(boots, dtype=float)
    finite = boots[np.isfinite(boots)]
    return {
        "delta": float(point_delta) if np.isfinite(point_delta) else float("nan"),
        "matched_value": float(point_matched) if np.isfinite(point_matched) else float("nan"),
        "point_clip": mean_clip,
        "point_psnr": mean_psnr,
        "lo": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
        "hi": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
        "out_of_range": bool(out_of_range),
        "n_boot_valid": int(finite.size),
        "n_images": int(n),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p2-config", default="configs/p2_howfar.yaml")
    parser.add_argument("--out-dir", default="results/p2")
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p2 = load_yaml(ROOT / args.p2_config)
    out_dir = ROOT / args.out_dir
    frame = pd.read_csv(out_dir / "per_image.csv")
    frame["image_id"] = frame["image_id"].astype(str).str.zfill(12)
    print(f"per_image.csv: {len(frame)} rows, {frame.image_id.nunique()} images", flush=True)

    n_boot = int(p2.get("bootstrap", N_BOOT_DEFAULT))
    t_main = 0.9
    raw = frame[frame.prox == 0]

    # ------------------------------------------------------------------ #
    # summary.csv: mean + CI per configuration
    # ------------------------------------------------------------------ #
    metric_cols = ["psnr", "lpips", "mse", "ssim", "clip_whole", "clip_edited", "struct_dist"]
    summary_rows: List[Dict] = []
    for (t, seed, method, param, prox), group in frame.groupby(
            ["t", "seed", "method", "param", "prox"], sort=True):
        row = {"t": t, "seed": seed, "method": method, "param": param, "prox": prox,
               "nfe": int(group["nfe"].iloc[0]), "n": len(group)}
        for column in metric_cols:
            ci = boot_ci(group[column].to_numpy(), n_boot=200)
            row[f"{column}_mean"] = ci["point"]
            row[f"{column}_lo"] = ci["lo"]
            row[f"{column}_hi"] = ci["hi"]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"summary.csv: {len(summary)} configurations", flush=True)

    # ------------------------------------------------------------------ #
    # per-image matrices for the matched comparisons
    # ------------------------------------------------------------------ #
    def matrix(t: float, seed: int, method: str, param: str, prox: int, column: str,
               index: pd.Index) -> np.ndarray:
        sub = frame[(frame.t == t) & (frame.seed == seed) & (frame.method == method)
                    & (frame.param == param) & (frame.prox == prox)]
        return sub.set_index("image_id")[column].reindex(index).to_numpy(dtype=float)

    def matrices_for(t: float, seed: int, prox: int) -> Tuple[pd.Index, Dict, Dict]:
        index = pd.Index(sorted(frame.image_id.unique()))
        clip, psnr = {}, {}
        lam_params = [f"{v:g}" for v in p2["lam_list"]]
        clip["lam"] = np.stack([matrix(t, seed, "lam", p, prox, "clip_edited", index)
                                for p in lam_params], axis=1)
        psnr["lam"] = np.stack([matrix(t, seed, "lam", p, prox, "psnr", index)
                                for p in lam_params], axis=1)
        for k in p2["picard_k"]:
            key = f"picard_{k}"
            clip[key] = matrix(t, seed, "picard", str(k), prox, "clip_edited", index)
            psnr[key] = matrix(t, seed, "picard", str(k), prox, "psnr", index)
        for k in p2["avg_k"]:
            for lam in p2["avg_lambda"]:
                key = f"avg_{k}_{lam:g}"
                clip[key] = matrix(t, seed, "avg", f"{k}_{lam:g}", prox, "clip_edited", index)
                psnr[key] = matrix(t, seed, "avg", f"{k}_{lam:g}", prox, "psnr", index)
        return index, clip, psnr

    index, clip_raw, psnr_raw = matrices_for(t_main, 0, 0)
    _, clip_prox, psnr_prox = matrices_for(t_main, 0, 1)

    # ΔLPIPS / ΔStructDist: same matched construction, only the metric arrays differ
    matched_rows: List[Dict] = []
    for prox_flag, clip_map, psnr_map in ((0, clip_raw, psnr_raw), (1, clip_prox, psnr_prox)):
        for metric_name in ("psnr", "lpips", "struct_dist"):
            lam_vals = []
            for p in [f"{v:g}" for v in p2["lam_list"]]:
                sub = frame[(frame.t == t_main) & (frame.seed == 0) & (frame.method == "lam")
                            & (frame.param == p) & (frame.prox == prox_flag)]
                lam_vals.append(sub.set_index("image_id")[metric_name].reindex(index).to_numpy(float))
            lam_metric = np.stack(lam_vals, axis=1)
            lam_clip_vals = np.stack(
                [frame[(frame.t == t_main) & (frame.seed == 0) & (frame.method == "lam")
                       & (frame.param == p) & (frame.prox == prox_flag)]
                 .set_index("image_id")["clip_edited"].reindex(index).to_numpy(float)
                 for p in [f"{v:g}" for v in p2["lam_list"]]], axis=1)
            targets = [f"picard_{k}" for k in p2["picard_k"] if k in (2, 3)]
            targets += [f"avg_{k}_{lam:g}" for k in p2["avg_k"] for lam in p2["avg_lambda"][:1]]
            for key in targets:
                method, param = ("picard", key.split("_")[1]) if key.startswith("picard") \
                    else ("avg", "_".join(key.split("_")[1:]))
                sub = frame[(frame.t == t_main) & (frame.seed == 0) & (frame.method == method)
                            & (frame.param == param) & (frame.prox == prox_flag)]
                if sub.empty:
                    continue
                point_metric = sub.set_index("image_id")[metric_name].reindex(index).to_numpy(float)
                point_clip = sub.set_index("image_id")["clip_edited"].reindex(index).to_numpy(float)
                result = pareto_delta(point_clip, point_metric, lam_clip_vals, lam_metric,
                                      n_boot=n_boot)
                matched_rows.append({"prox": prox_flag, "metric": metric_name, "variant": key,
                                     "mean": result["delta"], "lo": result["lo"],
                                     "hi": result["hi"], "out_of_range": result["out_of_range"],
                                     "point_clip": result["point_clip"],
                                     "point_value": result["point_psnr"],
                                     "matched_value": result["matched_value"],
                                     "n_images": result["n_images"]})
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(out_dir / "matched_clip.csv", index=False)

    # ------------------------------------------------------------------ #
    # H2a (CLIP gain K=3 vs K=1), H2b (Pareto), H2c (convergence), H2d (t)
    # ------------------------------------------------------------------ #
    h2a = paired_boot_ci(clip_raw["picard_3"] - clip_raw["picard_1"], n_boot=n_boot)
    h2a["wilcoxon_greater"] = wilcoxon_p(clip_raw["picard_3"] - clip_raw["picard_1"],
                                         alternative="greater")
    h2a["verdict"] = "SUPPORTED" if h2a["lo"] > 0 else "INCONCLUSIVE"
    h2a["delta_in_k1"] = float(np.nanmean(matrix(t_main, 0, "picard", "1", 0, "delta_in", index)))
    h2a["delta_in_k3"] = float(np.nanmean(matrix(t_main, 0, "picard", "3", 0, "delta_in", index)))

    h2b = {}
    for key in ("picard_2", "picard_3"):
        h2b[key] = pareto_delta(clip_raw[key], psnr_raw[key], clip_raw["lam"], psnr_raw["lam"],
                                n_boot=n_boot)
    supported = all(h2b[k]["delta"] > 0 and h2b[k]["lo"] > 0 and not h2b[k]["out_of_range"]
                    for k in h2b)
    refuted = all(h2b[k]["delta"] < 0 and h2b[k]["hi"] < 0 and not h2b[k]["out_of_range"]
                  for k in h2b)
    h2b["verdict"] = "SUPPORTED" if supported else ("REFUTED" if refuted else "INCONCLUSIVE")

    diag_rows = []
    for path in sorted((out_dir / "shards").glob("*_diag.csv")):
        diag_rows.extend(read_rows(path))
    diag = pd.DataFrame(diag_rows)
    for column in ("t", "seed", "k", "delta_in", "delta_out", "delta_ring", "rho_2_6",
                   "norm_k", "c_k", "diverged"):
        diag[column] = pd.to_numeric(diag[column], errors="coerce")
    diag.to_csv(out_dir / "diag.csv", index=False)

    conv = diag[(diag.k >= 2) & diag.c_k.notna()]
    h2c = {
        "c_median_by_k": {int(k): float(np.nanmedian(g["c_k"]))
                          for k, g in conv.groupby("k")},
        "c_iqr_by_k": {int(k): [float(np.nanpercentile(g["c_k"], 25)),
                                float(np.nanpercentile(g["c_k"], 75))]
                       for k, g in conv.groupby("k")},
        "diverged_fraction": float(diag.groupby(["image_id", "t", "seed"])["diverged"]
                                   .max().mean()),
        "diverged_fraction_by_category": {
            str(cat): float(g.groupby(["image_id", "t", "seed"])["diverged"].max().mean())
            for cat, g in diag.groupby("category")},
        "n_runs": int(diag.groupby(["image_id", "t", "seed"]).ngroups),
        "verdict": "REPORTED (no pre-registered decision rule)",
    }

    # H1c: does the halo shrink under Picard?  rho of delta_K vs K=1
    picard_halo = diag[diag.k >= 1][["image_id", "category", "t", "seed", "k", "rho_2_6"]]
    picard_halo.to_csv(out_dir.parent / "p1" / "picard_halo.csv", index=False)
    pivot = picard_halo[(picard_halo.t == t_main) & (picard_halo.seed == 0)].pivot_table(
        index="image_id", columns="k", values="rho_2_6")
    h1c = {"t": t_main, "seed": 0,
           "rho_median_by_k": {int(k): float(np.nanmedian(pivot[k])) for k in pivot.columns},
           "note": "H1c: median(rho_K=4 - rho_K=1) > 0 means the halo fades under Picard"}
    if 1 in pivot.columns and 4 in pivot.columns:
        diff = (pivot[4] - pivot[1]).dropna().to_numpy()
        h1c.update({"median_rho4_minus_rho1": float(np.median(diff)),
                    "ci": paired_boot_ci(diff, n_boot=n_boot),
                    "wilcoxon_greater": wilcoxon_p(diff, alternative="greater"),
                    "n": int(len(diff))})
        h1c["verdict"] = "SUPPORTED" if (h1c["median_rho4_minus_rho1"] > 0
                                         and h1c["wilcoxon_greater"]["p"] < 0.01) else "INCONCLUSIVE"
    else:
        h1c["verdict"] = "INCONCLUSIVE"

    # H2d: t=0.7 vs t=0.9
    _, clip_07, psnr_07 = matrices_for(0.7, 0, 0)
    d09 = pareto_delta(clip_raw["picard_3"], psnr_raw["picard_3"], clip_raw["lam"],
                       psnr_raw["lam"], n_boot=n_boot)
    d07 = pareto_delta(clip_07["picard_3"], psnr_07["picard_3"], clip_07["lam"],
                       psnr_07["lam"], n_boot=n_boot)
    h2d = {"delta_psnr_t0.9": d09, "delta_psnr_t0.7": d07}
    if not d09["out_of_range"] and not d07["out_of_range"]:
        diff = d07["delta"] - d09["delta"]
        h2d["comparison"] = "delta_psnr"
        h2d["diff"] = diff
        h2d["verdict"] = "SUPPORTED" if (d07["lo"] > d09["hi"]) else "INCONCLUSIVE"
    else:
        gain09 = float(np.nanmean(clip_raw["picard_3"] - clip_raw["picard_1"]))
        gain07 = float(np.nanmean(clip_07["picard_3"] - clip_07["picard_1"]))
        h2d.update({"comparison": "clip_gain_out_of_range", "gain_t0.9": gain09,
                    "gain_t0.7": gain07,
                    "verdict": "SUPPORTED" if gain07 > gain09 else "INCONCLUSIVE"})

    hypotheses = {"H2a": h2a, "H2b": h2b, "H2c": h2c, "H2d": h2d, "H1c": h1c}
    write_json(out_dir / "hypotheses.json", hypotheses)
    write_json(out_dir.parent / "p1" / "h1c.json", h1c)

    print(f"H1c: {h1c['verdict']}  median(rho4-rho1)={h1c.get('median_rho4_minus_rho1')}")
    print(f"H2a: {h2a['verdict']}  CLIP gain K3-K1 = {h2a['point']:+.3f} "
          f"CI [{h2a['lo']:+.3f},{h2a['hi']:+.3f}]")
    print(f"H2b: {h2b['verdict']}  " + "  ".join(
        f"{k}: dPSNR={h2b[k]['delta']:+.3f} CI[{h2b[k]['lo']:+.3f},{h2b[k]['hi']:+.3f}]"
        f"{' OUT' if h2b[k]['out_of_range'] else ''}" for k in ("picard_2", "picard_3")))
    print(f"H2c: diverged={h2c['diverged_fraction']:.4f}  c_median={h2c['c_median_by_k']}")
    print(f"H2d: {h2d['verdict']}  {h2d.get('comparison')}")

    make_figures(out_dir, summary, clip_raw, psnr_raw, clip_prox, psnr_prox, diag, p2, t_main)
    return 0


def make_figures(out_dir, summary, clip_raw, psnr_raw, clip_prox, psnr_prox, diag, p2, t_main) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for prox_flag, clip_map, psnr_map, tag in ((0, clip_raw, psnr_raw, "raw"),
                                               (1, clip_prox, psnr_prox, "prox")):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        lam = summary[(summary.t == t_main) & (summary.seed == 0) & (summary.method == "lam")
                      & (summary.prox == prox_flag)].sort_values("param", key=lambda s:
                                                                 s.astype(float))
        ax.plot(lam.clip_edited_mean, lam.psnr_mean, "o-", label="lam (sweep)", color="C0")
        for _, row in lam.iterrows():
            ax.annotate(f"{float(row['param']):g}", (row.clip_edited_mean, row.psnr_mean),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        pic = summary[(summary.t == t_main) & (summary.seed == 0) & (summary.method == "picard")
                      & (summary.prox == prox_flag)].sort_values("param", key=lambda s:
                                                                 s.astype(int))
        ax.plot(pic.clip_edited_mean, pic.psnr_mean, "s--", label="picard (K=1..6)", color="C3")
        for _, row in pic.iterrows():
            ax.annotate(f"K{row['param']}", (row.clip_edited_mean, row.psnr_mean), fontsize=6,
                        xytext=(3, -8), textcoords="offset points")
        avg = summary[(summary.t == t_main) & (summary.seed == 0) & (summary.method == "avg")
                      & (summary.prox == prox_flag)]
        if not avg.empty:
            ax.scatter(avg.clip_edited_mean, avg.psnr_mean, marker="^", label="avg", color="C2")
        recon = summary[(summary.method == "recon") & (summary.prox == prox_flag)]
        if not recon.empty:
            ax.axhline(recon.psnr_mean.iloc[0], color="grey", ls=":", lw=1,
                       label="recon (VAE ceiling)")
        ax.set_xlabel("CLIP-Edited (higher = better)")
        ax.set_ylabel("PSNR on the unedited part (dB)")
        ax.set_title(f"Pareto, t={t_main}, seed=0, {tag}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"pareto_t{t_main}_{tag}.png", dpi=160)
        plt.close(fig)

    conv = diag[(diag.k >= 2) & diag.c_k.notna()]
    if not conv.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        med = conv.groupby("k")["c_k"].median()
        q25 = conv.groupby("k")["c_k"].quantile(0.25)
        q75 = conv.groupby("k")["c_k"].quantile(0.75)
        axes[0].fill_between(med.index, q25, q75, alpha=0.3)
        axes[0].plot(med.index, med, "o-")
        axes[0].axhline(1.0, color="k", ls="--", lw=0.8)
        axes[0].set_xlabel("Picard step n")
        axes[0].set_ylabel("contraction ratio c_n")
        axes[0].set_title("Picard contraction (median, IQR)")
        axes[0].grid(alpha=0.3)
        cats = sorted(diag.category.unique(), key=lambda c: int(c))
        frac = [float(diag[diag.category == c].groupby(["image_id", "t", "seed"])["diverged"]
                      .max().mean()) for c in cats]
        axes[1].bar([str(c) for c in cats], frac)
        axes[1].set_xlabel("editing category")
        axes[1].set_ylabel("fraction of runs flagged diverged")
        axes[1].set_title("divergence by category")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_convergence.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
