"""P3b analysis: source-consistent noise (H3e) + antithetic noise, descriptive.

Reuses the P3a anchors: H3e is judged at the **same** two CLIP anchors as H3b (V0's mean
CLIP-Edited at lambda = 0.75 and 1.0, t = 0.9), taken from ``results/p3/a/hypotheses.json``
because P3b's own lambda grid (0.5, 1, 1.5, 2) does not contain 0.75.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import stats  # noqa: E402
from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402

T_MAIN = 0.9


def curves_for(frame: pd.DataFrame, t: float, kind: str) -> Dict[str, Dict[str, np.ndarray]]:
    sub = frame[(frame["t"] == t) & (frame["kind"] == kind)]
    ids = sorted(sub["image_id"].astype(str).unique())
    lams = sorted(float(v) for v in sub["lam"].unique())
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for J, block in sub.groupby("J"):
        rows = {}
        for value in ("clip_target_edit", "psnr"):
            piv = block.pivot_table(index="lam", columns="image_id", values=value, aggfunc="first")
            rows[value] = piv.reindex(index=lams, columns=ids).to_numpy(dtype=float)
        out[f"J{int(J)}"] = {"clip": rows["clip_target_edit"], "y": rows["psnr"]}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p3-config", default="configs/p3.yaml")
    parser.add_argument("--bootstrap", type=int, default=None)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / args.p3_config)
    n_boot = int(args.bootstrap or p3["bootstrap"])
    out = ROOT / base["output_root"] / "p3" / "b"
    frames = [pd.read_csv(p, dtype={"image_id": str, "category": str}, on_bad_lines="skip")
              for p in sorted((out / "shards").glob("shard[0-9]*.csv"))
              if not p.name.endswith("_diag.csv") and p.stat().st_size > 0]
    if not frames:
        raise SystemExit("no P3b shard CSVs yet")
    per = pd.concat(frames, ignore_index=True)
    per = per.drop_duplicates(subset=["image_id", "t", "kind", "J", "lam"])
    per.to_csv(out / "per_image.csv", index=False)

    diag_files = [pd.read_csv(p, dtype={"image_id": str, "category": str}, on_bad_lines="skip")
                  for p in sorted((out / "shards").glob("shard*_diag.csv")) if p.stat().st_size > 0]
    diag = pd.concat(diag_files, ignore_index=True).drop_duplicates(subset=["image_id", "t", "j"])
    diag.to_csv(out / "fixed_point_diag.csv", index=False)

    excluded = set(str(c) for c in p3["exclude_categories"])
    main_df = per[~per["category"].astype(str).isin(excluded)]

    # ---- anchors: take them from P3a (same t, same seed 0, same residuals) ----
    a_json = ROOT / base["output_root"] / "p3" / "a" / "hypotheses.json"
    anchors: List[float] = []
    anchor_source = "P3a V0 mean CLIP-Edited (lambda = 0.75, 1)"
    if a_json.exists():
        blk = json.load(a_json.open())["per_t"][f"{T_MAIN:g}"]["anchors"]
        anchors = [float(blk[k]["clip"]) for k in sorted(blk, key=float)]
    curves = curves_for(main_df, T_MAIN, "fixed_point")
    if not anchors:
        clip = np.nanmean(curves["J0"]["clip"], axis=1)
        lams = sorted(float(v) for v in main_df[(main_df["t"] == T_MAIN)]["lam"].unique())
        anchors = [float(clip[lams.index(1.0)])]
        anchor_source = "P3b J=0 mean CLIP-Edited at lambda = 1 (P3a json missing)"

    report: Dict = {"created": now(), "n_images": int(main_df["image_id"].nunique()),
                    "anchors": anchors, "anchor_source": anchor_source, "n_boot": n_boot}

    names = [f"J{int(j)}" for j in sorted({int(v) for v in main_df["J"].unique()}) if int(j) >= 0]
    h3e = []
    for anchor in anchors:
        res = stats.boot_psnr_at_clip(curves, names, anchor, n_boot=n_boot,
                                      rng=np.random.default_rng(0))
        base_draws = res["J0"]["draws"]
        entry = {"anchor_clip": anchor, "psnr_at_anchor": {k: v["psnr"] for k, v in res.items()},
                 "out_of_range": {k: v["out_of_range"] for k, v in res.items()}, "delta": {}}
        for name in names:
            draws = res[name]["draws"]
            length = min(draws.size, base_draws.size)
            if length == 0:
                entry["delta"][name] = {"point": float("nan"), "lo": float("nan"),
                                        "hi": float("nan"), "n": 0}
                continue
            diff = draws[:length] - base_draws[:length]
            entry["delta"][name] = {"point": float(diff.mean()),
                                    "lo": float(np.quantile(diff, 0.025)),
                                    "hi": float(np.quantile(diff, 0.975)), "n": int(length)}
        h3e.append(entry)

    # H3e: R^(8) vs R^(0) at both anchors, CI > 0 at both
    deltas8 = [entry["delta"].get("J8", {}) for entry in h3e]
    oor = [entry["out_of_range"].get("J8", True) for entry in h3e]
    if any(oor):
        status = "NOT_TESTABLE"
    elif all(d.get("lo", -1) > 0 for d in deltas8):
        status = "SUPPORTED"
    elif all(d.get("hi", 1) < 0 for d in deltas8):
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    report["H3e"] = {"rule": "CI of dPSNR(R^(8) - R^(0)) > 0 at both anchors (t=0.9)",
                     "status": status, "detail": h3e}

    # descriptive: convergence of the fixed-point iteration
    fp = diag[(diag["t"] == T_MAIN) & (diag["j"] >= 0)]
    if not fp.empty:
        e0 = fp[fp["j"] == 0].set_index("image_id")["e_src"]
        jmax = int(fp["j"].max())
        eJ = fp[fp["j"] == jmax].set_index("image_id")["e_src"]
        joined = pd.concat({"e0": e0, "eJ": eJ}, axis=1).dropna()
        report["fixed_point"] = {
            "j_max": jmax,
            "median_e_by_j": {str(int(j)): float(b["e_src"].median())
                              for j, b in fp.groupby("j")},
            "converged_frac": float((joined["eJ"] < 0.5 * joined["e0"]).mean()) if len(joined) else float("nan"),
            "median_step_last": float(fp[fp["j"] == jmax - 1]["step"].median()),
            "median_eps_kurt_last": float(fp[fp["j"] == jmax]["eps_kurt"].median()),
        }
        # the §2.1 identity, recorded as j = -1 rows
        ident = diag[diag["j"] == -1]["e_src"]
        report["identity_max_abs"] = float(ident.max()) if len(ident) else float("nan")

    # descriptive: antithetic noise vs seed-0/1 noise
    if (main_df["kind"] == "antithetic").any():
        anti = curves_for(main_df, T_MAIN, "antithetic")
        res = stats.boot_psnr_at_clip({**curves, **anti}, ["J0", "antithetic"],
                                     anchors[0], n_boot=n_boot, rng=np.random.default_rng(1))
        report["antithetic"] = {"anchor_clip": anchors[0],
                                "psnr": {k: v["psnr"] for k, v in res.items()},
                                "note": "descriptive; R(-eps_0) uses 2 UNet forwards like R(eps_0)"}

    # sds vs dds (descriptive): the A direction
    sds_curves = curves_for(main_df, T_MAIN, "sds")
    res = stats.boot_psnr_at_clip(sds_curves, sorted(sds_curves), anchors[0],
                                 n_boot=n_boot, rng=np.random.default_rng(2))
    report["sds_vs_dds"] = {"anchor_clip": anchors[0],
                            "psnr": {k: v["psnr"] for k, v in res.items()},
                            "note": "descriptive: A = x_hat_tar - x_s vs R = x_hat_tar - x_hat_src"}

    write_json(out / "hypotheses.json", report)
    print(f"[p3b analyse] H3e={status} n={report['n_images']} anchors={[round(a, 3) for a in anchors]}")
    print(f"[p3b analyse] wrote {out / 'hypotheses.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
