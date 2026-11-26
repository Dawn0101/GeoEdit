"""P3d-D2 analysis: is the model's likelihood-ratio map a usable verifier?

D2a: median AUROC (continuous) of the signed ``dl``, ``|dl|`` and ``l(x_s)`` against the GT
latent mask, for K in {2, 8}, plus a paired Wilcoxon ``|dl| > signed``.

D2b: within-image Spearman of ``L_i`` with CLIP-Edited / CLIP-D, and the gain of ``argmax L``
over seed 0 (paired image-level bootstrap), next to the random and off-mask-energy references.
Also the rank-sum combination of ``L`` and the off-mask energy, as the handoff asks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

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
    d2 = ROOT / base["output_root"] / "p3d" / "d2"

    px = pd.concat([pd.read_csv(p, dtype={"image_id": str, "category": str}, on_bad_lines="skip")
                    for p in sorted((d2 / "shards").glob("pixel_shard*.csv"))], ignore_index=True)
    px = px.drop_duplicates(subset=["image_id", "K"])
    px = px[~px["category"].astype(str).isin(excluded)]
    wh = pd.concat([pd.read_csv(p, dtype={"image_id": str, "category": str}, on_bad_lines="skip")
                    for p in sorted((d2 / "shards").glob("whole_shard*.csv"))], ignore_index=True)
    wh = wh.drop_duplicates(subset=["image_id", "seed"])
    wh = wh[~wh["category"].astype(str).isin(excluded)]
    print(f"[d2 analyse] pixel {len(px)} rows / {px.image_id.nunique()} images, "
          f"whole {len(wh)} rows / {wh.image_id.nunique()} images")

    report: Dict = {"created": now(), "n_images_d2a": int(px.image_id.nunique()),
                    "n_images_d2b": int(wh.image_id.nunique()), "n_boot": n_boot}

    # ---- D2a ----
    d2a = {}
    for K, block in px.groupby("K"):
        entry = {k: {"median": float(block[k].median()),
                     "mean": float(block[k].mean()),
                     "n": int(len(block))} for k in ("auroc_signed", "auroc_abs", "auroc_lsrc")}
        diff = (block["auroc_abs"] - block["auroc_signed"]).dropna()
        entry["abs_vs_signed_wilcoxon_p"] = float(
            scipy_stats.wilcoxon(diff, alternative="greater").pvalue) if len(diff) > 10 else float("nan")
        d2a[str(int(K))] = entry
    report["D2a"] = {"rule": "continuous AUROC against mask64; DiffEdit-style reference = 0.858 (t=0.5)",
                     "by_K": d2a}

    # ---- D2b ----
    rows: List[Dict] = []
    for image_id, block in wh.groupby("image_id"):
        if len(block) < 4:
            continue
        for target in ("clip_target_edit", "clip_d"):
            rho = scipy_stats.spearmanr(block["L"], block[target]).statistic
            if np.isfinite(rho):
                rows.append({"image_id": image_id, "target": target, "rho": float(rho)})
    sp = pd.DataFrame(rows)
    report["D2b"] = {"spearman": {}}
    for target, block in sp.groupby("target"):
        ci = paired_boot_ci(block["rho"].values, n_boot=n_boot, rng=np.random.default_rng(0))
        report["D2b"]["spearman"][target] = {"mean": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
                                             "frac_positive": float((block["rho"] > 0).mean()),
                                             "n": int(len(block))}

    # selection gains: argmax L, random, off-mask energy, and the rank-sum combination
    a_root = ROOT / base["output_root"] / "p3" / "a"
    piv = wh.pivot_table(index="image_id", columns="seed", values=["L", "clip_target_edit", "clip_d", "psnr"])
    off_energy = {}
    for image_id in piv.index:
        bundle = torch.load(a_root / "R" / f"{image_id}.pt", map_location="cpu", weights_only=True)
        R5 = torch.stack([bundle[f"t0.5_s{s}"] for s in range(4)], 0).float()
        R9 = torch.stack([bundle[f"t0.9_s{s}"] for s in range(4)], 0).float()
        e5 = R5.norm(dim=1).mean(dim=0)
        M5 = e5 / e5.max().clamp_min(1e-12)
        off_energy[image_id] = ((R9 * (1.0 - M5).unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=(1, 2, 3)).numpy()

    picks = {"argmax_L": {}, "offmask_energy": {}, "random": {}}
    rng = np.random.default_rng(0)
    for image_id in piv.index:
        L = piv.loc[image_id]["L"].values.astype(float)
        off = off_energy[image_id]
        picks["argmax_L"][image_id] = int(np.argmax(L))
        picks["offmask_energy"][image_id] = int(np.argmin(off))
        picks["random"][image_id] = int(rng.integers(0, 4))
        # rank-sum: low off-mask energy and high L are both good
        rank_L = scipy_stats.rankdata(-L)
        rank_E = scipy_stats.rankdata(off)
        picks.setdefault("rank_sum", {})[image_id] = int(np.argmin(rank_L + rank_E))

    gains = {}
    for rule, table in picks.items():
        entry = {}
        for metric in ("clip_target_edit", "clip_d", "psnr"):
            diffs = []
            for image_id, pick in table.items():
                diffs.append(piv.loc[image_id][(metric, pick)] - piv.loc[image_id][(metric, 0)])
            ci = paired_boot_ci(np.asarray(diffs, dtype=float), n_boot=n_boot, rng=np.random.default_rng(1))
            entry[metric] = {"point": ci["point"], "lo": ci["lo"], "hi": ci["hi"], "n": ci["n"]}
        gains[rule] = entry
    report["D2b"]["selection_gain_vs_seed0"] = gains
    pass_bar = (gains["argmax_L"]["clip_target_edit"]["point"] >= 0.3
                and gains["argmax_L"]["clip_target_edit"]["lo"] > 0) or \
               (gains["argmax_L"]["clip_d"]["point"] >= 0.005
                and gains["argmax_L"]["clip_d"]["lo"] > 0)
    report["D2b"]["status"] = "PASSED" if pass_bar else "FAILED"
    report["D2b"]["rule"] = ("dCLIP-Edited >= +0.3 or dCLIP-D >= +0.005, CI excluding 0 "
                             "(reference: CLIP oracle +1.58, random ~0)")

    table = pd.DataFrame([{"K": k, **{m: v[m]["median"] for m in ("auroc_signed", "auroc_abs", "auroc_lsrc")}}
                          for k, v in d2a.items()])
    table.to_csv(d2 / "p3d_d2_table.csv", index=False)
    write_json(d2 / "d2_hypotheses.json", report)
    print(f"[d2 analyse] D2a abs AUROC: " + ", ".join(f"K={k}: {v['auroc_abs']['median']:.3f}"
                                                     for k, v in d2a.items()))
    print(f"[d2 analyse] D2b {report['D2b']['status']}: "
          f"argmaxL dCLIP={gains['argmax_L']['clip_target_edit']['point']:+.3f}"
          f"[{gains['argmax_L']['clip_target_edit']['lo']:+.3f},"
          f"{gains['argmax_L']['clip_target_edit']['hi']:+.3f}] "
          f"dCLIPD={gains['argmax_L']['clip_d']['point']:+.4f}"
          f"[{gains['argmax_L']['clip_d']['lo']:+.4f},{gains['argmax_L']['clip_d']['hi']:+.4f}]")
    print(f"[d2 analyse] wrote {d2 / 'd2_hypotheses.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
