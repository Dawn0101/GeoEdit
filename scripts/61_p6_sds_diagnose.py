"""P6 Stage 0: is the in-mask SDS/attraction term ``A`` anything other than noise?

Before spending a single image on the k-sweep, the term has to justify itself.  Two things decide it:

1. **How big is the source-side error?**  ``A = R + e_src``, so ``||e_src||`` relative to ``||R||``
   says how far the attraction is from the plain residual.  If ``e_src`` is tiny, ``k > 0`` is a
   rescale of ``R`` and the whole exercise is re-parameterisation.
2. **Is ``A`` more repeatable than ``R``?**  The variance decomposition over the four t_h noise
   seeds splits each stack into a systematic part ``S`` (what repeats across draws -- the real edit
   direction) and a random part ``V`` (what does not).  ``k > 0`` only helps if ``A`` has a *higher*
   systematic share than ``R``, especially inside the mask; if it is worse, the term is injecting
   noise into the edit region and the pre-registered expectation is "k>0 does not improve".

No UNet forward: everything comes from ``results/p3/a/{R,xs}`` and ``results/p6/a/musrc``.

Outputs ``results/p6/report/p6_sds_diagnose.{csv,json}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.data import PIEDataset  # noqa: E402
from geoedit.edits import attention_term, l5_mask  # noqa: E402
from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402
from geoedit.masks import auroc, moments  # noqa: E402

SEEDS = (0, 1, 2, 3)


def decomp(stack: torch.Tensor, region: torch.Tensor) -> Dict[str, float]:
    """Systematic vs random energy of a ``[N,4,64,64]`` stack, inside/outside ``region``."""
    mean, s2 = moments(stack)
    m_sq = (mean ** 2).sum(dim=0)
    n = stack.shape[0]
    out = {}
    for tag, sel in (("in", region > 0.5), ("out", region <= 0.5)):
        if not bool(sel.any()):
            out[f"V_{tag}"] = out[f"S_{tag}"] = out[f"frac_random_{tag}"] = float("nan")
            continue
        v = float((4.0 * s2[sel]).mean())
        s = float((m_sq[sel] - 4.0 * s2[sel] / n).mean())
        out[f"V_{tag}"] = v
        out[f"S_{tag}"] = s
        out[f"frac_random_{tag}"] = v / (v + max(s, 0.0)) if (v + max(s, 0.0)) > 0 else float("nan")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p6-config", default="configs/p6.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    p6 = load_yaml(ROOT / args.p6_config)

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    a_root = ROOT / base["output_root"] / "p3" / "a"
    mu_root = ROOT / base["output_root"] / "p6" / "a" / "musrc"
    out_dir = ROOT / base["output_root"] / "p6" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    skipped = 0
    indices = range(len(dataset)) if not args.limit else range(min(args.limit, len(dataset)))
    for index in indices:
        sample = dataset[index]
        mu_path = mu_root / f"{sample.image_id}.pt"
        if not mu_path.exists():
            skipped += 1
            continue
        x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt", weights_only=True).float()
        x_s = x_s[0] if x_s.dim() == 4 else x_s  # cached as [1,4,64,64] -> [4,64,64]
        bundle = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
        mu = torch.load(mu_path, weights_only=True)
        R = torch.stack([bundle[f"t0.9_s{s}"].float() for s in SEEDS], 0)
        e = torch.stack([mu[f"t0.9_s{s}"].float() - x_s for s in SEEDS], 0)
        A = R + e

        M = l5_mask(bundle["t0.5_s0"].float() if "t0.5_s0" in bundle else bundle["t0.5_s0"])
        region = torch.from_numpy(sample.mask64).float()

        d_r = decomp(R, region)
        d_a = decomp(A, region)
        r_norm = float(R[0].norm())
        row = {
            "image_id": sample.image_id, "category": sample.category,
            "x_norm": float(x_s.norm()), "R_norm": r_norm,
            "e_norm": float(e[0].norm()), "A_norm": float(A[0].norm()),
            "e_over_x": float(e[0].norm() / x_s.norm().clamp_min(1e-12)),
            "e_over_R": float(e[0].norm() / max(r_norm, 1e-12)),
            "cos_A_R": float((A[0] * R[0]).sum() / (A[0].norm() * R[0].norm() + 1e-12)),
            "cos_e_R": float((e[0] * R[0]).sum() / (e[0].norm() * R[0].norm() + 1e-12)),
            "mask_area": float((M > 0.5).float().mean()),
            "gt_area": float(region.mean()),
        }
        for tag in ("in", "out"):
            for name, d in (("R", d_r), ("A", d_a)):
                row[f"{name}_S_{tag}"] = d[f"S_{tag}"]
                row[f"{name}_V_{tag}"] = d[f"V_{tag}"]
                row[f"{name}_frac_random_{tag}"] = d[f"frac_random_{tag}"]
        row["S_in_ratio_A_over_R"] = (row["A_S_in"] / row["R_S_in"]) if row["R_S_in"] > 0 else float("nan")
        # Does the *systematic* part localise?  The mean over seeds is the systematic estimate;
        # its energy map is what a mask would use.  R's version is the DiffEdit energy (P5: 0.858).
        mean_R = R.mean(dim=0)
        mean_A = A.mean(dim=0)
        row["auroc_meanR"] = auroc(mean_R.norm(dim=0).numpy(), region.numpy())
        row["auroc_meanA"] = auroc(mean_A.norm(dim=0).numpy(), region.numpy())
        row["auroc_passA"] = auroc(A[0].norm(dim=0).numpy(), region.numpy())
        row["snr_in_R"] = row["R_S_in"] / max(row["R_V_in"], 1e-12)
        row["snr_in_A"] = row["A_S_in"] / max(row["A_V_in"], 1e-12)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "p6_sds_diagnose.csv", index=False)
    if frame.empty:
        print("[diagnose] no images (run scripts/60 first)")
        return 1
    if skipped:
        print(f"[diagnose] skipped {skipped} images without mu_src")

    q = frame[["e_over_R", "e_over_x", "cos_A_R", "cos_e_R", "R_frac_random_in",
               "A_frac_random_in", "R_frac_random_out", "A_frac_random_out",
               "S_in_ratio_A_over_R", "snr_in_R", "snr_in_A",
               "auroc_meanR", "auroc_meanA", "auroc_passA"]].median()
    print(f"[diagnose] {len(frame)} images; medians:")
    for k, v in q.items():
        print(f"   {k:22s} {v:+.4f}")

    gate = p6["sds_gate"]
    # 2026-11-27 (DEVIATIONS P6-D2): the pre-registered check was `||e_src||/||R|| <= 1` ("stop if
    # the source error is large"), which is backwards: a *large* e_src means A is a genuinely
    # different direction, while a *small* one would mean A = R and k a mere rescale.  The
    # substantive criterion is repeatability, and that one is unchanged from the pre-registration.
    degenerate = bool(q["e_over_R"] < float(gate["degenerate_if_e_src_over_R_below"]))
    repeat_ok = bool(q["A_frac_random_in"] < q["R_frac_random_in"]) if \
        gate.get("require_frac_random_not_worse", True) else True
    e_ok = not degenerate
    verdict = {
        "created": now(), "n_images": int(len(frame)),
        "medians": {k: float(v) for k, v in q.items()},
        "gate": {"degenerate_if_e_src_over_R_below": gate["degenerate_if_e_src_over_R_below"],
                 "require_frac_random_not_worse": gate.get("require_frac_random_not_worse", True)},
        "A_is_a_different_direction": e_ok,
        "A_more_repeatable_in_mask": repeat_ok,
        "proceed_to_k_sweep": bool(e_ok and repeat_ok),
        "reading": "If A is not more repeatable than R inside the mask, the pre-registered expectation "
                   "is that k>0 does not improve the edit; the visual k-sweep still runs (the PI asked "
                   "to see it) but the frontier stops at k=1.",
        "note": "frac_random = V/(V+S) with V = mean(4 s^2) and S = mean(||mu||^2 - 4 s^2 / N) over the "
                "four t_h seeds; lower = more repeatable = better.",
    }
    write_json(out_dir / "p6_sds_diagnose.json", verdict)
    print(f"\n  ||e_src||/||R|| = {q['e_over_R']:.3f}  (degenerate if < "
          f"{gate['degenerate_if_e_src_over_R_below']}) -> {'A is a new direction' if e_ok else 'A ~ R'}")
    print(f"  AUROC of the systematic energy: mean_R {q['auroc_meanR']:.4f} "
          f"mean_A {q['auroc_meanA']:.4f} (single draw A {q['auroc_passA']:.4f})")
    print(f"  frac_random in mask: R {q['R_frac_random_in']:.4f} vs A {q['A_frac_random_in']:.4f} "
          f"-> {'A more repeatable' if repeat_ok else 'A LESS repeatable (k injects noise)'}")
    print(f"  proceed to k-sweep: {verdict['proceed_to_k_sweep']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
