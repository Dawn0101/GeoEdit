"""P6: the same-source A/B judge --- validity checks and pair scoring.

``--mode validity``   builds the four controls **on the fly** from the cached residuals and runs the
                      five pre-registered checks (``configs/p6.yaml`` §ab_validity):

                      C1 ``(V0, V0)``            identical pair -> the score must be ~0
                      C2 order antisymmetry       ``s_first`` vs ``-s_second``
                      C3 ``(V2, E1b)``            a real edit against same-envelope noise -> A wins
                      C4 ``(E1b, V2)``            the same pair reversed -> B wins
                      C5 ``(V0, E1c)``            **same amplitude, opposite direction** -> A wins

                      C5 is the check the P5 absolute-rating gate could not pass; if it fails here the
                      A/B form is not a usable semantic axis either and P6 stops.

``--mode pairs``      scores ``--a <dir>`` vs ``--b <dir>`` (one ``<image_id>.png`` per image in each),
                      which is how the matched-background-PSNR comparison (S3) is run.

Both modes score the two show orders and report ``s = mean(s_first, s_second)`` (positive = A follows
the instruction better).  A/B labels are assigned by ``sha256(image_id)`` parity so "A" is not
correlated with any variant.

Outputs ``results/p6/ab/{validity,<tag>}.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import edit_ops  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.editreward import EditRewardJudge  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402
from geoedit.vlm_judge import QwenYesNoJudge  # noqa: E402

FIELDS = ["image_id", "category", "pair", "a_name", "b_name", "s", "s_first", "s_second",
          "a_swapped", "psnr_a", "psnr_b", "instruction"]


def ab_swap(image_id: str, salt: str = "") -> bool:
    """Deterministic label assignment so "A" is never systematically one variant."""
    h = hashlib.sha256(f"{image_id}|{salt}".encode()).digest()
    return bool(h[0] & 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", choices=["validity", "pairs"], default="validity")
    parser.add_argument("--a", default=None,
                        help="run A image dir(s) for --mode pairs; comma separated for several pairs")
    parser.add_argument("--b", default=None, help="run B image dir(s), same order as --a")
    parser.add_argument("--tag", default=None, help="output tag (default: mode or A-vs-B names)")
    parser.add_argument("--n", type=int, default=100, help="images for --mode validity")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--judge-batch", type=int, default=3)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--instrument", choices=["diff", "ab"], default="diff",
                        help="diff = paired difference of the validated Yes/No score (no A/B "
                             "position degree of freedom); ab = forced choice logP(A)-logP(B)")
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = [i for i in range(len(dataset))]
    eval_idx = [pool[i] for i in np.linspace(0, len(pool) - 1, 200).round().astype(int).tolist()]
    if args.mode == "validity":
        picked = [i for k, i in enumerate(eval_idx[: args.n]) if k % args.num_shards == args.shard]
    else:
        dirs_a = [Path(x) for x in str(args.a).split(",")]
        dirs_b = [Path(x) for x in str(args.b).split(",")]
        if len(dirs_a) != len(dirs_b):
            raise SystemExit("--a and --b must list the same number of directories")
        pair_dirs = [(f"{a.name.replace('_psnr', '@')}_vs_{b.name.replace('_psnr', '@')}", a, b)
                     for a, b in zip(dirs_a, dirs_b)]
        ids = sorted(set.intersection(*[{p.stem for p in d.glob("*.png")}
                                        for _, a, b in pair_dirs for d in (a, b)]))
        by_id = {dataset[i].image_id: i for i in pool}
        picked = [by_id[i] for i in ids if i in by_id]
        picked = [i for k, i in enumerate(picked) if k % args.num_shards == args.shard]

    tag = args.tag or (("validity" if args.instrument == "ab" else "validity_diff")
                       if args.mode == "validity" else
                       f"{Path(args.a).name}_vs_{Path(args.b).name}")
    out_dir = ROOT / base["output_root"] / "p6" / "ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{tag}.csv"
    done = set()
    if args.only_missing and csv_path.exists():
        with csv_path.open() as fh:
            done = {(r["image_id"], r["pair"]) for r in pd.read_csv(csv_path).to_dict("records")}

    er = EditRewardJudge(device=args.device)
    judge = QwenYesNoJudge(device=args.device, backbone=er)  # shares the 7B backbone
    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision="fp32",
                    text_precision="fp32", vae_only=True)
    a_root = ROOT / base["output_root"] / "p3" / "a"
    print(f"[ab {tag}] {len(picked)} images on {args.device} (mode={args.mode})", flush=True)

    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(picked):
        sample = dataset[index]
        images: Dict[str, Image.Image] = {}
        psnr: Dict[str, float] = {}
        if args.mode == "validity":
            x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt",
                             weights_only=True).float().to(model.device)
            x_s = x_s[0] if x_s.dim() == 4 else x_s
            bundle = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
            R0 = bundle["t0.9_s0"].float().to(model.device)
            mask = torch.from_numpy(sample.mask64).float().to(model.device)
            g = torch.Generator(device="cpu").manual_seed(edit_ops.image_seed(sample.image_id, 777))
            xi = torch.randn(R0.shape, generator=g, dtype=torch.float32).to(model.device)
            env = R0.abs() * xi
            noise = env / env.norm().clamp_min(1e-12) * float(R0.norm())
            lat = {"V0": x_s + args.lam * R0,
                   "V2": x_s + args.lam * mask.unsqueeze(0) * R0,
                   "E1b": x_s + args.lam * noise,
                   "E1c": x_s - args.lam * R0}
            for name, z in lat.items():
                images[name] = Image.fromarray(model.decode(z.unsqueeze(0))[0])
            pairs = [("C1_identical", "V0", "V0"), ("C3_v2_vs_noise", "V2", "E1b"),
                     ("C4_noise_vs_v2", "E1b", "V2"), ("C5_v0_vs_e1c", "V0", "E1c")]
        else:
            images = {}
            pairs = []
            for pair_name, dir_a, dir_b in pair_dirs:
                images[(pair_name, "A")] = Image.open(dir_a / f"{sample.image_id}.png").convert("RGB")
                images[(pair_name, "B")] = Image.open(dir_b / f"{sample.image_id}.png").convert("RGB")
                pairs.append((pair_name, (pair_name, "A"), (pair_name, "B")))

        # background PSNR of every candidate (same definition as the official psnr_unedit)
        ref = np.asarray(sample.image(), dtype=np.float32)
        bg = sample.mask512 == 0
        for name, img in images.items():
            arr = np.asarray(img, dtype=np.float32)
            psnr[name] = float(10 * np.log10(255.0 ** 2 /
                                             max(float(((arr - ref)[bg] ** 2).mean()), 1e-12)))

        rows = []
        for pair_name, a_name, b_name in pairs:
            if (sample.image_id, pair_name) in done:
                continue
            img_a, img_b = images[a_name], images[b_name]
            if args.instrument == "diff":
                # Paired difference of the P5 instrument: each candidate is scored against the
                # *same* source, so the "something changed" floor cancels in the difference and
                # there is no A/B position to bias.  Both show orders are computed and averaged.
                src = sample.image()
                fwd = judge.score_pairs([(src, img_a, sample.instruction),
                                         (src, img_b, sample.instruction)],
                                        batch_size=args.judge_batch)
                swp = judge.score_pairs([(src, img_a, sample.instruction),
                                         (src, img_b, sample.instruction)],
                                        order="swapped", batch_size=args.judge_batch)
                out = {"s_first": fwd[0] - fwd[1], "s_second": swp[0] - swp[1]}
                out["s"] = 0.5 * (out["s_first"] + out["s_second"])
            elif ab_swap(sample.image_id, pair_name):
                res = judge.score_ab([(img_b, img_a, sample.instruction)], batch_size=args.judge_batch)
                out = {"s": -res[0]["s"], "s_first": -res[0]["s_first"], "s_second": -res[0]["s_second"]}
            else:
                res = judge.score_ab([(img_a, img_b, sample.instruction)], batch_size=args.judge_batch)
                out = res[0]
            rows.append({"image_id": sample.image_id, "category": sample.category, "pair": pair_name,
                         "a_name": a_name, "b_name": b_name, "s": out["s"],
                         "s_first": out["s_first"], "s_second": out["s_second"],
                         "a_swapped": float(ab_swap(sample.image_id, pair_name)),
                         "psnr_a": psnr.get(a_name, float("nan")),
                         "psnr_b": psnr.get(b_name, float("nan")),
                         "instruction": sample.instruction})
        if rows:
            append_rows(csv_path, rows, FIELDS)
        processed += 1
        if processed % 5 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(picked)} {el / processed:.1f}s/img", flush=True)

    print(f"[ab {tag}] done={processed} in {(time.perf_counter() - start) / 60:.1f} min ({now()})",
          flush=True)
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
