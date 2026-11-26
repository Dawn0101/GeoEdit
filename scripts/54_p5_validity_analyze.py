"""P5-§1.3: validity of the semantic judges (pre-registered checks, 540 images, lam=1).

For every judge ``s`` the gain of a variant ``v`` over the unedited source is

    G(v) = s(v) - s(recon)

and the four pre-registered checks (``docs/handoff/v4_p5.md`` §1.3) are

| check | pass condition |
|---|---|
| ``src_not_best``  | G(V0) > 0 and G(V2) > 0, CI excludes 0 |
| ``e1a_wrong``     | G(E1a) < 0.25 G(V0) |
| ``e1b_noise``     | G(E1b) < 0.25 G(V0) |
| ``e1c_reverse``   | G(E1c) <= 0 |

Judges: ``clip_d`` (the P4 CLIP-D, as the incumbent), ``er_dim1`` (EditReward head 0,
Instruction Following), ``q_follow`` (Qwen2.5-VL logP(Yes) - logP(No)).  Secondary:
``er_dim2`` (Visual Quality) and ``q_keep`` (preservation).

All intervals are image-level paired bootstraps (2000 draws).  ``q_follow``/``q_keep`` are also
re-run with the two images swapped (a fixed subset of images, ``q_*_sw``) to expose position bias.

Outputs ``results/p5/validity/p5_*.csv`` and ``p5_judge_verdict.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402
from geoedit.stats import boot_ci  # noqa: E402

JUDGES = {
    "clip_d": {"metric": "clip_d", "sign": +1, "label": "CLIP-D (P4)"},
    "er_dim1": {"metric": "er_dim1", "sign": +1, "label": "EditReward head0 (instruction following)"},
    "er_dim2": {"metric": "er_dim2", "sign": +1, "label": "EditReward head1 (visual quality)"},
    "q_follow": {"metric": "q_follow", "sign": +1, "label": "Qwen2.5-VL logP(Yes/No) following"},
    "q_keep": {"metric": "q_keep", "sign": +1, "label": "Qwen2.5-VL logP(Yes/No) preservation"},
}
PRIMARY = ["clip_d", "er_dim1", "q_follow"]
VARIANT_ORDER = ["recon", "V0", "V2", "E1a", "E1b", "E1c"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--tag", default="validity",
                        help="results/p5/<tag> (validity_lam050 for the post-hoc sensitivity run)")
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    n_boot = int(args.bootstrap)
    out_dir = ROOT / base["output_root"] / "p5" / args.tag

    frame = pd.concat([pd.read_csv(p, dtype={"image_id": str, "category": str}, on_bad_lines="skip")
                       for p in sorted((out_dir / "shards").glob("shard*.csv"))], ignore_index=True)
    frame = frame.drop_duplicates(subset=["image_id", "variant"])
    n_img = frame["image_id"].nunique()
    print(f"[validity] {len(frame)} rows, {n_img} images, variants={sorted(frame.variant.unique())}")

    wide = frame.pivot(index="image_id", columns="variant")
    rows, checks = [], []
    for judge, spec in JUDGES.items():
        col = spec["metric"]
        if (col, "recon") not in wide:
            continue
        recon = wide[(col, "recon")]
        stats = {}
        for v in VARIANT_ORDER:
            if (col, v) not in wide:
                continue
            value = wide[(col, v)]
            gain = value - recon
            ci = boot_ci(gain.values, n_boot=n_boot, rng=np.random.default_rng(0))
            s_ci = boot_ci(value.values, n_boot=n_boot, rng=np.random.default_rng(0))
            stats[v] = {"gain": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
                        "mean": s_ci["point"], "n": ci["n"]}
            rows.append({"judge": judge, "metric": col, "variant": v, "mean_score": s_ci["point"],
                         "gain": ci["point"], "gain_lo": ci["lo"], "gain_hi": ci["hi"],
                         "n": ci["n"]})
        if "V0" not in stats:
            continue
        g0 = stats["V0"]["gain"]
        ratio = {}
        for v in ("E1a", "E1b", "E1c"):
            if v in stats:
                ratio[v] = (stats[v]["gain"] / g0) if abs(g0) > 1e-12 else float("nan")

        def add(check: str, value: float, lo: float, hi: float, ok: bool, note: str = "") -> None:
            checks.append({"judge": judge, "metric": col, "check": check, "statistic": value,
                           "lo": lo, "hi": hi, "passed": bool(ok), "note": note})

        # 1) the unedited image must not be the best
        v0, v2 = stats.get("V0"), stats.get("V2")
        ok1 = bool(v0 and v2 and v0["lo"] > 0 and v2["lo"] > 0)
        add("src_not_best", float("nan"),
            min(v0["lo"] if v0 else np.nan, v2["lo"] if v2 else np.nan),
            min(v0["hi"] if v0 else np.nan, v2["hi"] if v2 else np.nan), ok1,
            f"G(V0)={v0['gain']:+.3f}[{v0['lo']:+.3f},{v0['hi']:+.3f}] "
            f"G(V2)={v2['gain']:+.3f}[{v2['lo']:+.3f},{v2['hi']:+.3f}]" if v0 and v2 else "")
        # 2/3) wrong prompt and pure noise must be far below V0
        for check, v in (("e1a_wrong", "E1a"), ("e1b_noise", "E1b")):
            if v not in stats:
                continue
            diff = wide[(col, v)] - wide[(col, "recon")] - 0.25 * (wide[(col, "V0")] - wide[(col, "recon")])
            ci = boot_ci(diff.values, n_boot=n_boot, rng=np.random.default_rng(1))
            add(check, stats[v]["gain"], ci["lo"], ci["hi"], bool(ci["hi"] < 0),
                f"G({v})/G(V0)={ratio.get(v, float('nan')):.2f} (threshold 0.25); "
                f"CI is for G({v}) - 0.25 G(V0)")
        # 4) the reversed edit must not gain
        if "E1c" in stats:
            diff = wide[(col, "E1c")] - wide[(col, "recon")]
            ci = boot_ci(diff.values, n_boot=n_boot, rng=np.random.default_rng(2))
            add("e1c_reverse", stats["E1c"]["gain"], ci["lo"], ci["hi"], bool(ci["hi"] <= 0), "")

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "p5_judge_validity.csv", index=False)
    check_table = pd.DataFrame(checks)
    check_table.to_csv(out_dir / "p5_judge_checks.csv", index=False)

    print("\nmean score and gain over the unedited source (paired bootstrap):")
    for judge in JUDGES:
        block = table[table.judge == judge]
        if block.empty:
            continue
        print(f"  [{JUDGES[judge]['label']}]")
        for _, r in block.iterrows():
            print(f"    {r['variant']:6s} score {r['mean_score']:+8.3f}  "
                  f"gain {r['gain']:+8.3f} [{r['gain_lo']:+.3f}, {r['gain_hi']:+.3f}]")

    print("\npre-registered checks:")
    print(check_table[["judge", "check", "statistic", "lo", "hi", "passed"]].round(3).to_string(index=False))

    verdict = {}
    for judge in JUDGES:
        block = check_table[check_table.judge == judge]
        if block.empty:
            continue
        verdict[judge] = {"passed": bool(block["passed"].all()),
                          "checks": block.set_index("check")["passed"].to_dict()}
    passed = [j for j in PRIMARY if verdict.get(j, {}).get("passed")]
    print(f"\njudges passing all four checks (primary set): {passed or 'NONE'}")

    # position bias on the swapped subset
    pos_rows = []
    for judge, kind in (("q_follow", "following"), ("q_keep", "preservation")):
        fwd, swp = f"{'q_follow' if kind == 'following' else 'q_keep'}", \
                   f"{'q_follow_sw' if kind == 'following' else 'q_keep_sw'}"
        sub = frame[frame.variant.isin(["V0", "V2", "recon"])]
        sub = sub[np.isfinite(sub[swp])]
        if sub.empty:
            continue
        agree = float(((sub[fwd] > 0) == (sub[swp] > 0)).mean())
        rho = spearmanr(sub[fwd], sub[swp]).statistic if len(sub) > 3 else float("nan")
        pos_rows.append({"judge": judge, "question": kind, "n": int(len(sub)),
                         "sign_agreement": agree, "spearman": float(rho),
                         "mean_forward": float(sub[fwd].mean()),
                         "mean_swapped": float(sub[swp].mean()),
                         "mean_shift": float((sub[swp] - sub[fwd]).mean())})
        shift = float((sub[swp] - sub[fwd]).mean())
        print(f"  position bias [{judge}]: n={len(sub)} sign agreement {agree:.3f}  "
              f"spearman {rho:+.3f}  mean shift {shift:+.3f}")
    pos = pd.DataFrame(pos_rows)
    if not pos.empty:
        pos.to_csv(out_dir / "p5_judge_position_bias.csv", index=False)

    write_json(out_dir / "p5_judge_verdict.json", {
        "created": now(), "n_images": int(n_img), "n_boot": n_boot, "lam": 1.0,
        "verdict": verdict, "primary_passing": passed,
        "verdict_reason": {
            "clip_d": "amplitude meter per P4 §2; re-checked here at matched lambda only for the "
                      "side-by-side table",
            "rule": "§1.3: a judge may be used as the semantic metric only if all four checks pass; "
                    "if both pass, Qwen is primary and EditReward secondary; if none passes, stop "
                    "and export 50 side-by-side examples for the PI.",
        },
        "position_bias": pos.to_dict(orient="records"),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
