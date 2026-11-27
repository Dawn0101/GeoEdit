"""P6 analysis: A/B validity (C1-C5), matched-PSNR lambda solving, and the main A/B table.

``--mode validity``  reads ``results/p6/ab/validity_lam*.csv`` and applies the five pre-registered
                    checks.  ``s`` is ``logP(A) - logP(B)`` averaged over the two show orders
                    (positive = A follows the instruction better).

| check | construction | pass |
|---|---|---|
| C1 | ``(V0, V0)`` identical pair | mean ``s`` ~ 0, CI contains 0 |
| C2 | order antisymmetry | ``spearman(s_first, -s_second) > 0.7`` and mean ``s_first+s_second`` ~ 0 |
| C3 | ``(V2, E1b)`` | ``s > 0``, CI excludes 0 |
| C4 | ``(E1b, V2)`` | ``s < 0``, CI excludes 0 |
| C5 | ``(V0, E1c)`` same amplitude, opposite direction | ``s > 0``, CI excludes 0 |

``--mode matched``   builds ``背景PSNR(lam)`` per run from ``results/p6/frontier/per_image.csv`` and
                    solves the lambda that hits each target on the **mean** curve (no extrapolation).

``--mode report``    turns a directory of ``ab/*.csv`` into the matched-PSNR table: paired
                    image-level bootstrap on ``s``, win rate, and the ``|ΔPSNR| < tol`` subset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402
from geoedit.stats import boot_ci  # noqa: E402


def _interp(x, y, target):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 or target < x.min() or target > x.max():
        return None
    order = np.argsort(x)
    return float(np.interp(target, x[order], y[order]))


def validity(out_dir: Path, n_boot: int) -> int:
    files = sorted(out_dir.glob("validity*.csv"))
    if not files:
        print("[validity] no files")
        return 1
    report: Dict[str, dict] = {"created": now(), "n_boot": n_boot, "lam": {}}
    for path in files:
        lam = path.stem.replace("validity_diff_lam0", "0.").replace("validity_lam0", "0.")
        frame = pd.read_csv(path, dtype={"image_id": str, "category": str})
        frame = frame.drop_duplicates(subset=["image_id", "pair"])
        print(f"\n[{path.name}] {len(frame)} rows, {frame.image_id.nunique()} images")
        res = {}
        for pair, block in frame.groupby("pair"):
            ci = boot_ci(block["s"].values, n_boot=n_boot, rng=np.random.default_rng(0))
            # "diff" instrument: both orders measure the same sign, so agreement is rho(s_first,
            # s_second) > 0.  "ab" instrument: the orders are antisymmetric by construction and the
            # check was rho(s_first, -s_second).
            rho, _ = spearmanr(block["s_first"], block["s_second"])
            bias = boot_ci((block["s_first"] + block["s_second"]).values, n_boot=n_boot,
                           rng=np.random.default_rng(1))
            win = float((block["s"] > 0).mean())
            res[pair] = {"n": int(len(block)), "mean": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
                         "win_rate": win, "order_rho": float(rho),
                         "order_shift_mean": bias["point"], "order_shift_lo": bias["lo"],
                         "order_shift_hi": bias["hi"]}
            print(f"  {pair:16s} s = {ci['point']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]  "
                  f"win {win:.2f}  order_rho {rho:+.2f}")

        def passed(name):
            return name in res

        checks = {}
        if passed("C1_identical"):
            c = res["C1_identical"]
            checks["C1"] = bool(c["lo"] <= 0 <= c["hi"] and abs(c["mean"]) < 0.30)
        if passed("C3_v2_vs_noise"):
            checks["C3"] = bool(res["C3_v2_vs_noise"]["lo"] > 0)
        if passed("C4_noise_vs_v2"):
            checks["C4"] = bool(res["C4_noise_vs_v2"]["hi"] < 0)
        if passed("C5_v0_vs_e1c"):
            checks["C5"] = bool(res["C5_v0_vs_e1c"]["lo"] > 0)
        rhos = [v["order_rho"] for v in res.values() if np.isfinite(v["order_rho"])]
        checks["C2"] = bool(rhos and min(rhos) > 0.70) if "diff" not in path.stem else \
            bool(rhos and min(rhos) > 0.30)
        print("  checks:", {k: ("PASS" if v else "FAIL") for k, v in checks.items()})
        report["lam"][lam] = {"pairs": res, "checks": checks,
                              "all_pass": bool(all(checks.values()) and len(checks) == 5)}
    write_json(out_dir / "p6_ab_validity.json", report)
    primary = report["lam"].get("0.5") or list(report["lam"].values())[0]
    print(f"\nPRIMARY (lam=0.5) all pass: {primary['all_pass']}  -> checks {primary['checks']}")
    return 0


def matched(out_dir: Path, frontier: Path, targets: List[float]) -> int:
    frame = pd.read_csv(frontier, dtype={"image_id": str})
    curve = (frame.groupby(["run", "lam"])[["psnr_unedit"]].mean().reset_index())
    pivot = curve.pivot(index="run", columns="lam", values="psnr_unedit")
    pivot.to_csv(out_dir / "p6_curves_psnr.csv")
    print(f"[matched] {len(pivot)} runs, lams {list(pivot.columns)}")
    print(pivot.round(2).to_string())
    rows = []
    for run, series in pivot.iterrows():
        xs = [float(c) for c in pivot.columns]
        ys = [float(series[c]) for c in pivot.columns]
        for t in targets:
            lam = _interp(ys, xs, t)  # psnr -> lam
            if lam is None:
                continue
            rows.append({"run": run, "target": t, "lam": lam})
    spec = pd.DataFrame(rows)
    spec.to_csv(out_dir / "matched_lam.csv", index=False)
    print(f"\n[matched] wrote {len(spec)} (run, target, lam) rows")
    print(spec.pivot(index="run", columns="target", values="lam").round(3).to_string())
    return 0


def report_mode(out_dir: Path, n_boot: int, tol: float) -> int:
    files = sorted(p for p in out_dir.glob("*.csv")
                   if not p.name.startswith(("validity", "p6_", "matched_")))
    if not files:
        print("[report] no pair files")
        return 1
    rows = []
    for path in files:
        frame = pd.read_csv(path, dtype={"image_id": str, "category": str})
        if "pair" not in frame.columns or "image_id" not in frame.columns:
            continue
        frame = frame.drop_duplicates(subset=["image_id", "pair"])
        for pair, block in frame.groupby("pair"):
            if pair.startswith("C"):
                continue
            for subset, sel in (("all", np.ones(len(block), bool)),
                                (f"|dPSNR|<{tol:g}", (block["psnr_a"] - block["psnr_b"]).abs() < tol)):
                values = block.loc[sel, "s"].values
                if values.size < 20:
                    continue
                ci = boot_ci(values, n_boot=n_boot, rng=np.random.default_rng(0))
                rows.append({"file": path.stem, "pair": pair, "subset": subset, "n": int(values.size),
                             "mean_s": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
                             "win_rate": float((values > 0).mean()),
                             "mean_dpsnr": float((block.loc[sel, "psnr_a"] -
                                                  block.loc[sel, "psnr_b"]).mean())})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "p6_ab_table.csv", index=False)
    print(table.round(3).to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", choices=["validity", "matched", "report"], default="validity")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--tol", type=float, default=1.0)
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    p6 = load_yaml(ROOT / "configs/p6.yaml")
    out_dir = ROOT / base["output_root"] / "p6" / "ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "validity":
        return validity(out_dir, int(args.bootstrap))
    if args.mode == "matched":
        targets = ([float(t) for t in args.targets.split(",")] if args.targets
                   else [float(t) for t in p6["matched_psnr"]["targets"]])
        return matched(out_dir, ROOT / base["output_root"] / "p6" / "frontier" / "per_image.csv",
                       targets)
    return report_mode(out_dir, int(args.bootstrap), float(args.tol))


if __name__ == "__main__":
    sys.exit(main())
