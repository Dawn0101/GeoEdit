"""P3a analysis: descriptive statistics (§3.2) + pre-registered judgements (§3.3).

Reads the shard CSVs written by ``30_p3a_build_and_eval.py`` and produces

``results/p3/a/per_image.csv``        merged rows (one per image, t, variant, lambda)
``results/p3/a/mask_quality.csv``     merged mask quality (AUROC / IoU vs mask64)
``results/p3/a/summary.csv``          mean / se per (subset, t, variant, lambda)
``results/p3/a/variance_decomp.csv``  V, S, frac_random in / out of the edit mask, per image
``results/p3/a/hypotheses.json``      H3a-H3d, the gap-closure ratio G, and the extras
``results/p3/a/fig_*.png``            one Pareto figure per noise level, plus two diagnostics

Judgements are *not* allowed to look at anything that was not pre-registered.  Every
decision below carries the hypothesis id from ``docs/handoff/v3_p3.md`` in a comment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import masks, stats  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402

ODDS = ("SUPPORTED", "REFUTED", "INCONCLUSIVE")


def judge(los: List[float], his: Optional[List[float]] = None) -> str:
    """Pre-registered convention: CI entirely > 0 -> SUPPORTED, entirely < 0 -> REFUTED.

    A CI that straddles 0 is INCONCLUSIVE (P2's convention); ``his`` defaults to ``los``.
    """
    his = list(los) if his is None else list(his)
    if not los:
        return "INCONCLUSIVE"
    if all(v > 0 for v in los):
        return "SUPPORTED"
    if all(v < 0 for v in his):
        return "REFUTED"
    return "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# merge + grids
# --------------------------------------------------------------------------- #
def merge_shards(out: Path) -> pd.DataFrame:
    shards = sorted((out / "shards").glob("shard[0-9]*.csv"))
    shards = [p for p in shards if not p.name.endswith("_mask.csv")]
    frames = [pd.read_csv(p, on_bad_lines="skip", dtype={"image_id": str, "category": str})
              for p in shards if p.stat().st_size > 0]
    if not frames:
        raise SystemExit(f"no shard CSVs under {out / 'shards'}")
    per = pd.concat(frames, ignore_index=True)
    # a shard-count change between runs must never duplicate a row
    per = per.drop_duplicates(subset=["image_id", "t", "variant", "lam"])
    per = per.sort_values(["image_id", "t", "variant", "lam"])
    per.to_csv(out / "per_image.csv", index=False)

    mask_files = sorted((out / "shards").glob("shard[0-9]*_mask.csv"))
    mframes = [pd.read_csv(p, on_bad_lines="skip", dtype={"image_id": str, "category": str})
               for p in mask_files if p.stat().st_size > 0]
    mask_q = pd.concat(mframes, ignore_index=True)
    mask_q = mask_q.drop_duplicates(subset=["image_id", "t", "mask_type"])
    mask_q = mask_q.sort_values(["image_id", "t", "mask_type"])
    mask_q.to_csv(out / "mask_quality_shards.csv", index=False)
    return per


def grid_matrices(sub: pd.DataFrame, value: str) -> Tuple[List[float], List[str], Dict[str, np.ndarray]]:
    """``{variant: [n_lam, n_img]}`` with a shared image order (NaN where missing)."""
    ids = sorted(sub["image_id"].astype(str).unique())
    lams = sorted(float(v) for v in sub["lam"].unique())
    out: Dict[str, np.ndarray] = {}
    for variant in sorted(sub["variant"].unique()):
        block = sub[sub["variant"] == variant]
        piv = block.pivot_table(index="lam", columns="image_id", values=value, aggfunc="first")
        piv = piv.reindex(index=lams, columns=ids)
        out[variant] = piv.to_numpy(dtype=float)
    return lams, ids, out


def curves_for_t(per: pd.DataFrame, t: float) -> Tuple[List[float], List[str], Dict]:
    sub = per[per["t"] == t]
    lams, ids, clip = grid_matrices(sub, "clip_target_edit")
    _, _, psnr = grid_matrices(sub, "psnr")
    return lams, ids, {v: {"clip": clip[v], "y": psnr[v]} for v in clip}



def _strip_draws(result: Dict) -> Dict:
    """Drop the raw bootstrap draws before serialising (they belong in memory, not in a JSON)."""
    return {name: {k: v for k, v in entry.items() if k != "draws"}
            for name, entry in result.items()}

def summary_table(per: pd.DataFrame, subsets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    metrics = ["psnr", "lpips", "mse", "ssim", "struct_dist",
               "clip_target_whole", "clip_target_edit", "clip_source_whole"]
    rows = []
    for name, frame in subsets.items():
        grouped = frame.groupby(["t", "variant", "lam"])
        for (t, variant, lam), group in grouped:
            row = {"subset": name, "t": t, "variant": variant, "lam": lam, "n": len(group)}
            for metric in metrics:
                values = group[metric].astype(float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_se"] = float(values.std(ddof=1) / np.sqrt(max(len(values), 1)))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["subset", "t", "variant", "lam"])


# --------------------------------------------------------------------------- #
# variance decomposition (needs the fp32 residual bundles)
# --------------------------------------------------------------------------- #
def variance_decomposition(out: Path, dataset: PIEDataset, t_list: List[float],
                           ids: List[str], include_categories: Optional[List[str]]) -> pd.DataFrame:
    rows = []
    for image_id in ids:
        path = out / "R" / f"{image_id}.pt"
        if not path.exists():
            continue
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        sample = dataset[image_id]
        if include_categories is not None and sample.category in include_categories:
            continue
        mask64 = torch.from_numpy(sample.mask64).float()
        outside = 1.0 - mask64
        for t in t_list:
            Rs = torch.stack([bundle[f"t{t:g}_s{s}"] for s in (0, 1, 2, 3)], dim=0).float()
            for region_name, region in (("in", mask64), ("out", outside)):
                dec = masks.variance_decomp(Rs, region)
                rows.append({"image_id": image_id, "category": sample.category, "t": t,
                             "region": region_name, **dec})
    return pd.DataFrame(rows)


def mask_areas(out: Path, ids: List[str], t_list: List[float], c_main: float, pool: int,
               dataset=None) -> pd.DataFrame:
    """Mask size + mask quality, recomputed from the bundles.

    ``auroc`` uses the **continuous** score, as §2.1 requires; the per-shard numbers thresholded
    first, which makes AUROC meaningless (it can only take a few values) -- see DEVIATIONS.
    """
    rows = []
    for image_id in ids:
        path = out / "R" / f"{image_id}.pt"
        if not path.exists():
            continue
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        gt = dataset[image_id].mask64 if dataset is not None else None
        for t in t_list:
            Rs = torch.stack([bundle[f"t{t:g}_s{s}"] for s in (0, 1, 2, 3)], dim=0).float()
            w4 = masks.snr_weight(Rs, c=c_main, pool=pool)[0]
            w2 = masks.snr_weight(Rs[:2], c=c_main, pool=pool)[0]
            d_cont = Rs.norm(dim=1).mean(dim=0) / Rs.norm(dim=1).mean(dim=0).max().clamp_min(1e-12)
            e_cont = Rs[0].norm(dim=0) / Rs[0].norm(dim=0).max().clamp_min(1e-12)
            # (§2.1) AUROC on the *continuous* score, IoU on the >0.5 mask
            pairs = {
                "w4": (w4, (w4 > 0.5).float()),
                "w2": (w2, (w2 > 0.5).float()),
                "diffedit": (d_cont, (d_cont > 0.5).float()),
                "energy": (e_cont, (e_cont > 0.5).float()),
            }
            if t == 0.9 and "t0.5_s0" in bundle:
                fine = torch.stack([bundle[f"t0.5_s{s}"] for s in (0, 1, 2, 3)], dim=0).float()
                ct = w4 * masks.snr_weight(fine, c=c_main, pool=pool)[0]
                pairs["ct"] = (ct, (ct > 0.5).float())
            for name, (cont, binary) in pairs.items():
                c_np, b_np = masks.to_numpy(cont), masks.to_numpy(binary)
                rows.append({"image_id": image_id, "t": t, "mask_type": name,
                             "area_frac": float(binary.mean()), "mean_w": float(cont.mean()),
                             "auroc": masks.auroc(c_np, gt) if gt is not None else float("nan"),
                             "iou": masks.iou(b_np, gt) if gt is not None else float("nan")})
    return pd.DataFrame(rows)



# --------------------------------------------------------------------------- #
# Post-hoc: the frontier in the other direction (matched background PSNR)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(out: Path, per: pd.DataFrame, var: pd.DataFrame, mask_q: pd.DataFrame,
                 t_list: List[float]) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    for t in t_list:
        sub = per[per["t"] == t]
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for variant, block in sub.groupby("variant"):
            means = block.groupby("lam")[["clip_target_edit", "psnr"]].mean().sort_index()
            ax.plot(means["clip_target_edit"], means["psnr"], marker="o", ms=3, lw=1.2,
                    label=variant)
        ax.set_xlabel("CLIP-Edited (target, edit part)")
        ax.set_ylabel("PSNR (background)")
        ax.set_title(f"P3a Pareto, t={t}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        path = out / f"fig_pareto_t{t:g}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(str(path.relative_to(ROOT)))

    if not var.empty:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        data, labels = [], []
        for t in sorted(var["t"].unique()):
            for region in ("in", "out"):
                block = var[(var["t"] == t) & (var["region"] == region)]["frac_random"].dropna()
                data.append(block.values)
                labels.append(f"t={t:g}\n{region}")
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_ylabel("frac_random = V / (S + V)")
        ax.set_title("P3a variance decomposition (latent space)")
        ax.grid(alpha=0.3, axis="y")
        path = out / "fig_variance_decomp.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(str(path.relative_to(ROOT)))

    if not mask_q.empty:
        main = mask_q[mask_q["t"] == 0.9].groupby("mask_type")[["auroc", "iou"]].median()
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        main.plot.bar(ax=ax)
        ax.set_ylabel("median vs mask64")
        ax.set_title("P3a mask quality (t=0.9)")
        ax.grid(alpha=0.3, axis="y")
        path = out / "fig_mask_quality.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(str(path.relative_to(ROOT)))
    return made


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p3-config", default="configs/p3.yaml")
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--skip-variance", action="store_true")
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / args.p3_config)
    out = ROOT / base["output_root"] / "p3" / "a"
    n_boot = int(args.bootstrap or p3["bootstrap"])
    exclude = [str(c) for c in p3["exclude_categories"]]
    t_main = 0.9

    per = merge_shards(out)
    per["category"] = per["category"].astype(str)
    main_df = per[~per["category"].isin(exclude)]
    print(f"[merge] {len(per)} rows, main subset {len(main_df)} rows, "
          f"{main_df['image_id'].nunique()} images", flush=True)

    subsets = {"main": main_df, "all": per}
    summary = summary_table(per, subsets)
    summary.to_csv(out / "summary.csv", index=False)

    mask_q_path = out / "mask_quality.csv"
    mask_q = pd.read_csv(mask_q_path, dtype={"image_id": str}) if mask_q_path.exists() \
        else pd.read_csv(out / "mask_quality_shards.csv", dtype={"image_id": str})
    if "category" not in mask_q.columns:
        mask_q["category"] = ""
    mask_q["category"] = mask_q["category"].astype(str)

    if not args.skip_variance:
        dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
        ids = sorted(main_df["image_id"].unique())
        var = variance_decomposition(out, dataset, [float(t) for t in p3["t_list"]], ids, None)
        var.to_csv(out / "variance_decomp.csv", index=False)
        areas = mask_areas(out, ids, [float(t) for t in p3["t_list"]],
                           float(p3["c_main"]), int(p3["pool"]), dataset=dataset)
        areas.to_csv(out / "mask_quality.csv", index=False)
        for name, block in areas.groupby("mask_type"):
            print(f"[mask] {name:9s} " + " ".join(
                f"t{t:g}: auroc={b.auroc.median():.3f} iou={b.iou.median():.3f} area={b.area_frac.median():.3f}"
                for t, b in block.groupby("t")), flush=True)
        print(f"[mask area] " + ", ".join(
            f"{k}={v:.3f}" for k, v in areas.groupby("mask_type").area_frac.median().items()), flush=True)
        print(f"[variance] {len(var)} rows", flush=True)
    else:
        var = (pd.read_csv(out / "variance_decomp.csv", dtype={"image_id": str, "category": str})
               if (out / "variance_decomp.csv").exists() else pd.DataFrame())

    t_list = [float(t) for t in p3["t_list"]]
    anchor_lams = [float(v) for v in p3["anchor_lambda"]]
    report: Dict = {"created": now(), "config": {"p3": p3, "n_boot": n_boot},
                    "n_images_main": int(main_df["image_id"].nunique()), "per_t": {}}

    # ---- per-t curve matching (H3b / H3c and the descriptive extras) ----
    for t in t_list:
        lams, ids, curves = curves_for_t(main_df, t)
        anchor_idx = [lams.index(v) for v in anchor_lams if v in lams]
        anchors = [lams[i] for i in anchor_idx]
        match = stats.mean_curve_match(curves, "V0", anchor_index=anchor_idx,
                                       n_boot=n_boot, rng=np.random.default_rng(0))
        block: Dict = {"anchors": {f"{a:g}": {"lambda": a,
                                              "clip": match["anchor_clip"][str(i)],
                                              "n_images": match["n"]}
                                   for i, a in zip(anchor_idx, anchors)},
                       "curve_match": {}}

        for i, a in zip(anchor_idx, anchors):
            column = {}
            for variant, cm in match["results"][str(i)].items():
                column[variant] = {"delta": cm.delta, "lo": cm.lo, "hi": cm.hi,
                                   "psnr_at_anchor": cm.psnr_at_anchor,
                                   "out_of_range": cm.out_of_range, "n_boot": cm.n}
            block["curve_match"][f"{a:g}"] = column

        def in_range(anchor_i: int, name: str) -> bool:
            cm = match["results"][str(anchor_i)].get(name)
            return bool(cm is not None and not cm.out_of_range)

        def paired(anchor_i: int, name_a: str, name_b: str) -> Dict[str, float]:
            # a matched-CLIP delta only means something if both curves actually reach the anchor
            if not (in_range(anchor_i, name_a) and in_range(anchor_i, name_b)):
                return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0,
                        "out_of_range": True}
            a_draws = np.asarray(match["boot"][str(anchor_i)].get(name_a) or [], dtype=float)
            b_draws = np.asarray(match["boot"][str(anchor_i)].get(name_b) or [], dtype=float)
            length = min(a_draws.size, b_draws.size)
            if length == 0:
                return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
            diff = a_draws[:length] - b_draws[:length]
            return {"point": float(diff.mean()),
                    "lo": float(np.quantile(diff, 0.025)),
                    "hi": float(np.quantile(diff, 0.975)),
                    "n": int(length), "out_of_range": False}

        # H3b (main): the SNR weight pushes the frontier out, at both anchors
        h3b = [paired(i, "V4", "V0") for i in anchor_idx]
        oor = [bool(d.get("out_of_range")) for d in h3b]
        block["H3b"] = {
            "rule": "CI of dPSNR(V4 - V0) > 0 at both anchors",
            "status": ("NOT_TESTABLE" if any(oor) else judge([d["lo"] for d in h3b], [d["hi"] for d in h3b])),
            "reason": ("designed anchors lie outside V4's CLIP range; the hypothesis cannot be "
                       "evaluated without extrapolation (post-hoc anchors are in posthoc.bracket)"
                       if any(oor) else ""),
            "detail": h3b}

        # H3c: better than the DiffEdit-style and the single-draw energy mask
        h3c_7 = [paired(i, "V4", "V7") for i in anchor_idx] if "V7" in curves else []
        h3c_8 = [paired(i, "V4", "V8") for i in anchor_idx] if "V8" in curves else []
        h3c_all = h3c_7 + h3c_8
        h3c_oor = any(bool(d.get("out_of_range")) for d in h3c_all)
        if h3c_7 and h3c_8 and not h3c_oor:
            both_support = (all(d["lo"] > 0 for d in h3c_7) and all(d["lo"] > 0 for d in h3c_8))
            any_refute = any(d["hi"] < 0 for d in h3c_all)
            h3c_status = "SUPPORTED" if both_support else ("REFUTED" if any_refute else "INCONCLUSIVE")
        else:
            h3c_status = "NOT_TESTABLE" if h3c_oor else "INCONCLUSIVE"
        block["H3c"] = {
            "rule": "CI of dPSNR(V4 - V7) and dPSNR(V4 - V8) both > 0; REFUTED if either CI < 0",
            "vs_V7": h3c_7, "vs_V8": h3c_8, "status": h3c_status,
            "reason": ("one of V4 / V7 / V8 cannot reach the designed anchor; not testable "
                       "without extrapolation" if h3c_oor else "")}

        # gap closure G = (V4 - V0) / (V2 - V0)  -- how much of the oracle gap V4 closes
        g_rows = []
        for i, a in zip(anchor_idx, anchors):
            abs_draws = {name: np.asarray(match["boot_abs"][str(i)].get(name) or [], dtype=float)
                         for name in ("V0", "V2", "V4")}
            length = min(v.size for v in abs_draws.values()) if abs_draws else 0
            if length:
                denom = abs_draws["V2"][:length] - abs_draws["V0"][:length]
                numer = abs_draws["V4"][:length] - abs_draws["V0"][:length]
                ok = np.abs(denom) > 1e-9
                ratio = numer[ok] / denom[ok]
                g_rows.append({"lambda": a, "n": int(ratio.size),
                               "point": float(np.median(ratio)) if ratio.size else float("nan"),
                               "lo": float(np.quantile(ratio, 0.025)) if ratio.size else float("nan"),
                               "hi": float(np.quantile(ratio, 0.975)) if ratio.size else float("nan")})
            else:
                g_rows.append({"lambda": a, "n": 0, "point": float("nan"),
                               "lo": float("nan"), "hi": float("nan")})
        block["gap_closure"] = g_rows

        # descriptive extras (no judgement)
        extras = {}
        for name in ("V1", "V5", "V6", "V10", "V11", "V4c05", "V4c2"):
            if name in curves:
                extras[f"{name}_vs_V4"] = [paired(i, name, "V4") for i in anchor_idx]
        block["extras"] = extras

        # post-hoc: matched-PSNR frontier + the same anchor test on the whole-image CLIP
        block["posthoc"] = {
            "note": ("not pre-registered. The masked variants cannot reach the pre-registered "
                     "CLIP anchors, so the frontier is also read at matched background PSNR."),
            "clip_at_psnr": stats.boot_clip_at_psnr(curves, sorted(curves), [25.0, 28.0, 31.0],
                                             n_boot=n_boot, rng=np.random.default_rng(2)),
        }
        # one bracket per hypothesis, over the variants that hypothesis compares
        for tag, members in (("H3b", ["V0", "V4"]), ("H3c", ["V0", "V4", "V7", "V8"])):
            bracket = stats.common_clip_range(curves, [m for m in members if m in curves])
            if bracket and bracket[0] < bracket[1]:
                lo, hi = bracket
                targets = [lo + (hi - lo) * f for f in (0.25, 0.5, 0.75)]
                block["posthoc"][f"bracket_{tag}"] = {
                    "common_clip_range": [lo, hi],
                    "members": [m for m in members if m in curves],
                    "note": ("anchors inside the CLIP range that every compared variant reaches; "
                             "post-hoc, because the pre-registered anchors sit above it"),
                    "anchors": {f"{c:.3f}": _strip_draws(
                        stats.boot_psnr_at_clip(curves, sorted(curves), c, n_boot=n_boot,
                                                rng=np.random.default_rng(4)))
                        for c in targets}}
            else:
                block["posthoc"][f"bracket_{tag}"] = {"common_clip_range": None,
                                                      "members": members}

        if "clip_target_whole" in per.columns:
            clip_w = grid_matrices(main_df[main_df["t"] == t], "clip_target_whole")[2]
            _, _, psnr_mats = grid_matrices(main_df[main_df["t"] == t], "psnr")
            curves_w = {v: {"clip": clip_w[v], "y": psnr_mats[v]} for v in clip_w}
            match_w = stats.mean_curve_match(curves_w, "V0", anchor_index=anchor_idx,
                                             n_boot=n_boot, rng=np.random.default_rng(3))
            block["posthoc"]["whole_clip"] = {
                f"{a:g}": {v: {"delta": c.delta, "lo": c.lo, "hi": c.hi,
                               "out_of_range": c.out_of_range}
                           for v, c in match_w["results"][str(i)].items()}
                for i, a in zip(anchor_idx, anchors)}

        # H3a: a single draw keeps more semantics than the mean (max over lambda, t=0.9)
        if t == t_main and "V2" in curves and "V3" in curves:
            h3a = stats.boot_mean_of_max(curves["V2"]["clip"], curves["V3"]["clip"],
                                         n_boot=n_boot, rng=np.random.default_rng(1))
            block["H3a"] = {"rule": "CI of max_lambda CLIP(V2) - max_lambda CLIP(V3) > 0",
                            "status": judge([h3a["lo"]], [h3a["hi"]]), "detail": h3a}
        report["per_t"][f"{t:g}"] = block
        print(f"[analyse] t={t:g} H3b={block['H3b']['status']} "
              f"H3c={block['H3c']['status']} " + (f"H3a={block['H3a']['status']} " if "H3a" in block else ""),
              flush=True)

    # ---- H3d: more of the random energy sits outside the mask than inside (t=0.9) ----
    if not var.empty:
        pivot = var[var["t"] == t_main].pivot_table(index="image_id", columns="region",
                                                    values="frac_random")
        if {"in", "out"}.issubset(pivot.columns):
            diff = (pivot["out"] - pivot["in"]).dropna()
            test = stats.wilcoxon_p(diff.values, alternative="greater")
            report["H3d"] = {
                "rule": "one-sided paired Wilcoxon of frac_random(out) - frac_random(in) > 0, p < 0.01",
                "status": ("SUPPORTED" if test["p"] < 0.01
                           else ("REFUTED" if test["p"] > 0.99 else "INCONCLUSIVE")),
                "detail": {"median_diff": float(np.median(diff.values)) if diff.size else float("nan"),
                           **test},
            }
            print(f"[analyse] H3d={report['H3d']['status']} p={test['p']:.3g}", flush=True)

    report["figures"] = make_figures(out, main_df, var, mask_q, t_list)
    write_json(out / "hypotheses.json", report)
    print(f"[analyse] wrote {out / 'hypotheses.json'} ({now()})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
