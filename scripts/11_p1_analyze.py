"""Stage B/C: turn the saved P1 residuals into rho / rho_null / e_rel and test H1, H1b.

Reads ``results/p1/latents/*.pt`` (computed by ``10_p1_compute.py``) and the per-shard row
tables, then writes

``results/p1/per_image.csv``   one row per (image, t, seed, sigma, band)
``results/p1/summary.json``    H1 / H1b verdicts with medians, CIs, p-values, n
``results/p1/fig_*.png``       the figures named in §9.1

H1c (the halo shrinking under Picard) needs the P2 iterates and is filled in by
``22_p2_analyze.py`` writing ``results/p1/picard_halo.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from geoedit import halo
from geoedit.data import PIEDataset
from geoedit.halo import e_rel, prepare_null, rho, rho_null_prepared
from geoedit.io_utils import append_rows, load_latent, load_yaml, read_rows, write_json
from geoedit.stats import boot_ci, paired_boot_ci, wilcoxon_p

ROOT = Path(__file__).resolve().parents[1]
FIELD_ORDER = ["image_id", "category", "t", "seed", "sigma", "band", "n_pix", "rho",
               "rho_null", "rho_minus_null", "e_rel", "excluded", "exclude_reason", "n_null"]


def band_label(band: Tuple[float, float]) -> str:
    lo, hi = band
    return f"{lo:g}_{hi:g}" if np.isfinite(hi) else f"{lo:g}_inf"


def load_rows(shard_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(shard_dir.glob("*_rows.csv")):
        rows.extend(read_rows(path))
    if not rows:
        raise SystemExit(f"no shard rows under {shard_dir}")
    return rows


def verify_grid(rows: List[Dict], t_list, seeds) -> Dict[str, int]:
    seen = {(r["image_id"], round(float(r["t"]), 3), int(r["seed"])) for r in rows}
    expected = len({r["image_id"] for r in rows}) * len(t_list) * len(seeds)
    return {"rows": len(rows), "unique_points": len(seen), "expected_points": expected}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p1-config", default="configs/p1_halo.yaml")
    parser.add_argument("--out-dir", default="results/p1")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p1 = load_yaml(ROOT / args.p1_config)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t_list = [float(t) for t in p1["t_list"]]
    seeds = [int(s) for s in p1["seeds"]]
    sigma_main = float(p1["sigma_main"])
    sigma_sec = float(p1["sigma_secondary"])
    bands = [tuple(float(v) for v in b) for b in p1["bands"]]
    main_band = tuple(float(v) for v in p1["main_band"])
    min_pix = int(p1["min_band_pixels"])
    max_interior = float(p1["max_interior_frac"])
    excluded_categories = {str(c) for c in p1["exclude_categories"]}

    dataset = PIEDataset(pie_root=Path(base["pie_root"]))
    rows = load_rows(ROOT / base["output_root"] / "p1" / "shards")
    print(f"rows: {len(rows)}  grid: {verify_grid(rows, t_list, seeds)}", flush=True)

    by_image: Dict[str, List[Dict]] = {}
    for row in rows:
        by_image.setdefault(row["image_id"], []).append(row)

    per_image: List[Dict] = []
    examples = []
    for count, (image_id, image_rows) in enumerate(sorted(by_image.items()), start=1):
        sample = dataset[image_id]
        prepared = prepare_null(sample.mask64, n_shifts=int(p1["null_shifts"]),
                                gap=float(p1["null_gap"]), rng=np.random.default_rng(0))
        for row in image_rows:
            t, seed = round(float(row["t"]), 3), int(row["seed"])
            latent = load_latent(ROOT / base["output_root"] / "p1" / "latents" /
                                 f"{image_id}_t{t:g}_s{seed}.pt")

            def evaluate(band, sigma, is_primary: bool) -> Dict:
                value, n_pix = rho(latent, sample.mask64, sample.dist64, band=band, sigma=sigma)
                null_value, n_null = rho_null_prepared(latent, prepared, band=band, sigma=sigma)
                reason = ""
                if sample.category in excluded_categories:
                    reason = f"category_{sample.category}"
                elif sample.interior_fraction > max_interior:
                    reason = "interior_frac_gt_half"
                elif n_pix < min_pix:
                    reason = f"band_pixels_lt_{min_pix}"
                elif not is_primary and sigma != sigma_main:
                    reason = "secondary_sigma"
                return {
                    "image_id": image_id, "category": sample.category, "t": t, "seed": seed,
                    "sigma": sigma, "band": band_label(band), "n_pix": n_pix,
                    "rho": value, "rho_null": null_value, "rho_minus_null": value - null_value,
                    "e_rel": (e_rel(latent, sample.mask64, sample.dist64, band=main_band)
                              if is_primary else np.nan),
                    "excluded": int(bool(reason)), "exclude_reason": reason, "n_null": n_null,
                }

            band_values = {}
            # The primary test band (2,6] is not necessarily in the curve list, so make sure
            # it is always evaluated at the primary sigma.
            curve_bands = list(bands) + ([main_band] if main_band not in bands else [])
            for band in curve_bands:
                entry = evaluate(band, sigma_main, is_primary=(band == main_band))
                band_values[band_label(band)] = entry["rho"]
                per_image.append(entry)
            per_image.append(evaluate(main_band, sigma_sec, is_primary=False))
            if t == 0.9 and seed == 0 and sample.category not in excluded_categories:
                examples.append((sample, latent, band_values))
        if count % 100 == 0:
            print(f"  analysed {count}/{len(by_image)} images", flush=True)

    append_rows(out_dir / "per_image.csv", per_image, FIELD_ORDER)
    print(f"per_image.csv: {len(per_image)} rows", flush=True)

    # ------------------------------------------------------------------ #
    # H1: t=0.9, main band, main sigma, seeds averaged per image
    # ------------------------------------------------------------------ #
    def collect(t: float, band: str, sigma: float) -> Dict[str, np.ndarray]:
        buckets: Dict[str, Dict[str, List[float]]] = {}
        for row in per_image:
            if row["sigma"] != sigma or row["band"] != band or row["t"] != t:
                continue
            if row["excluded"]:
                continue
            entry = buckets.setdefault(row["image_id"], {"rho": [], "null": [], "e": [], "cat": []})
            entry["rho"].append(row["rho"])
            entry["null"].append(row["rho_null"])
            entry["e"].append(row["e_rel"])
            entry["cat"].append(row["category"])
        return {
            "rho": np.array([np.nanmean(v["rho"]) for v in buckets.values()]),
            "null": np.array([np.nanmean(v["null"]) for v in buckets.values()]),
            "e_rel": np.array([np.nanmean(v["e"]) for v in buckets.values()]),
            "category": np.array([v["cat"][0] for v in buckets.values()]),
            "image_id": np.array(list(buckets.keys())),
        }

    main_label = band_label(main_band)
    h1_data = collect(0.9, main_label, sigma_main)
    diff = h1_data["rho"] - h1_data["null"]
    h1 = {
        "band": main_label, "sigma": sigma_main, "t": 0.9,
        "n_images": int(np.isfinite(h1_data["rho"]).sum()),
        "rho": boot_ci(h1_data["rho"]),
        "rho_null": boot_ci(h1_data["null"]),
        "rho_minus_null": boot_ci(diff),
        "rho_median": float(np.nanmedian(h1_data["rho"])),
        "rho_minus_null_median": float(np.nanmedian(diff)),
        "wilcoxon_rho": wilcoxon_p(h1_data["rho"], alternative="less"),
        "wilcoxon_diff": wilcoxon_p(diff, alternative="less"),
        "wilcoxon_diff_greater": wilcoxon_p(diff, alternative="greater"),
    }
    supported = (h1["rho_median"] < 0 and h1["wilcoxon_rho"]["p"] < 0.01
                 and h1["rho_minus_null_median"] < 0 and h1["wilcoxon_diff"]["p"] < 0.01)
    refuted = h1["rho_minus_null_median"] > 0 and h1["wilcoxon_diff_greater"]["p"] < 0.01
    h1["verdict"] = "SUPPORTED" if supported else ("REFUTED" if refuted else "INCONCLUSIVE")

    # H1b: e_rel median grows as t decreases
    e_by_t = {t: collect(t, main_label, sigma_main)["e_rel"] for t in t_list}
    diffs = {}
    ordered = sorted(t_list, reverse=True)  # 0.9, 0.7, 0.5
    for high, low in zip(ordered, ordered[1:]):
        common = min(len(e_by_t[high]), len(e_by_t[low]))
        diffs[f"{low:g}_minus_{high:g}"] = paired_boot_ci(e_by_t[low][:common] - e_by_t[high][:common])
    h1b_supported = all(d["lo"] > 0 for d in diffs.values()) and len(diffs) > 0
    h1b = {
        "e_rel_median_by_t": {f"{t:g}": float(np.nanmedian(e_by_t[t])) for t in ordered},
        "paired_diffs": diffs,
        "verdict": "SUPPORTED" if h1b_supported else "INCONCLUSIVE",
    }

    # per-band curve for the figure
    band_curve = {}
    for t in ordered:
        curve = {"rho": [], "null": []}
        for band in bands:
            data = collect(t, band_label(band), sigma_main)
            curve["rho"].append(float(np.nanmedian(data["rho"])) if data["rho"].size else float("nan"))
            curve["null"].append(float(np.nanmedian(data["null"])) if data["null"].size else float("nan"))
        band_curve[f"{t:g}"] = curve

    summary = {
        "grid": verify_grid(rows, t_list, seeds),
        "counts": {
            "rows_total": len(per_image),
            "rows_excluded": int(sum(r["excluded"] for r in per_image)),
            "per_image_rows_main": int(sum(1 for r in per_image if r["sigma"] == sigma_main)),
        },
        "H1": h1,
        "H1b": h1b,
        "H1c": {"verdict": "PENDING", "note": "filled by 22_p2_analyze.py from P2 Picard latents"},
        "band_curve": {t: band_curve[t] for t in band_curve},
        "bands": [band_label(b) for b in bands],
    }
    write_json(out_dir / "summary.json", summary)

    print(f"H1: {h1['verdict']}  median rho={h1['rho_median']:.4f}  "
          f"median(rho-null)={h1['rho_minus_null_median']:.4f}  n={h1['n_images']}", flush=True)
    print(f"H1b: {h1b['verdict']}  e_rel medians={h1b['e_rel_median_by_t']}", flush=True)

    make_figures(out_dir, per_image, band_curve, bands, ordered, examples, args.examples, p1)
    return 0


def make_figures(out_dir, per_image, band_curve, bands, t_order, examples, n_examples, p1) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [band_label(b) for b in bands]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for t in t_order:
        axes[0].plot(labels, band_curve[f"{t:g}"]["rho"], marker="o", label=f"rho t={t:g}")
        axes[1].plot(labels, band_curve[f"{t:g}"]["null"], marker="s", ls="--",
                     label=f"null t={t:g}")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("median rho")
    axes[0].set_title("residual vs blurred interior (negative = halo)")
    axes[1].set_ylabel("median rho_null")
    axes[1].set_title("mask-shift null")
    for ax in axes:
        ax.set_xlabel("distance band (latent cells)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_rho_vs_band.png", dpi=160)
    plt.close(fig)

    main_label = band_label(tuple(float(v) for v in p1["main_band"]))
    sel = [r for r in per_image if r["band"] == main_label and r["t"] == 0.9
           and r["sigma"] == float(p1["sigma_main"]) and not r["excluded"]]
    if sel:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist([r["rho"] for r in sel], bins=40, alpha=0.6, label="rho")
        ax.hist([r["rho_null"] for r in sel], bins=40, alpha=0.6, label="rho_null")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("rho (band 2-6, sigma 3, t=0.9)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fig_rho_hist_t0.9.png", dpi=160)
        plt.close(fig)

        rows_by_cat: Dict[str, List[float]] = {}
        for r in sel:
            rows_by_cat.setdefault(r["category"], []).append(r["rho_minus_null"])
        cats = sorted(rows_by_cat, key=int)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot([rows_by_cat[c] for c in cats], tick_labels=cats, showmeans=True)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("editing category")
        ax.set_ylabel("rho - rho_null")
        ax.set_title("halo on the band (2,6]; negative = counter-typed ring")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_per_category.png", dpi=160)
        plt.close(fig)

    if examples:
        picked = examples[:: max(1, len(examples) // n_examples)][:n_examples]
        cols = 4
        rows = int(np.ceil(len(picked) / cols))
        fig, axes = plt.subplots(rows, cols * 4, figsize=(cols * 9, rows * 2.6))
        axes = np.atleast_2d(axes)
        for i, (sample, latent, _) in enumerate(picked):
            r, c = divmod(i, cols)
            res = latent.float()
            norm_map = res[0].pow(2).sum(0).sqrt().numpy()
            mask_t = torch.from_numpy(sample.mask64)[None, None]
            u = halo.gaussian_blur(res * mask_t, float(p1["sigma_main"]))
            proj = (res * u).sum(1)[0].numpy()
            vmax = np.nanpercentile(np.abs(proj), 99) + 1e-8
            band = halo.band_mask(sample.dist64, tuple(float(v) for v in p1["main_band"]))
            panels = [
                (np.asarray(sample.image()), None, f"{sample.image_id} c{sample.category}"),
                (sample.mask64, None, "GT mask (64)"),
                (norm_map, "viridis", "|R|"),
                (proj, "coolwarm", "<R,u> (band 2-6 contour)"),
            ]
            for j, (data, cmap, title) in enumerate(panels):
                ax = axes[r, c * 4 + j]
                if data.ndim == 3:
                    ax.imshow(data)
                else:
                    ax.imshow(data, cmap=cmap, vmin=(-vmax if cmap == "coolwarm" else None),
                              vmax=(vmax if cmap == "coolwarm" else None))
                    if j == 3:
                        ax.contour(band.astype(float), levels=[0.5], colors="k", linewidths=0.6)
                ax.set_title(title, fontsize=7)
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_examples.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    with torch.no_grad():
        raise SystemExit(main())
