"""P5-§1.3 / §1.4: score the cached residuals with all three judges.

No UNet forward is needed for the frontier arms: every candidate is ``x_s + lam * delta`` with
``delta`` built from the P1 residuals (``results/p3/a/R``) and ``x_s`` (``results/p3/a/xs``).  Only
``E1a`` (the wrong-prompt control) needs two UNet forwards per image, and those are recomputed
exactly as in ``scripts/46_p4_metric_validity.py`` (same permutation seed, same category rule).

Two specs:

``validity``  (P5-§1.3) 540 images (categories 0-7), ``lam=1``, variants
              ``recon | V0 | V2 | E1a | E1b | E1c`` -- the four pre-registered controls plus the
              two references.  Every image also gets the swapped-order Qwen scores when its rank
              falls in the position-bias subset (``--swap-mod``).
``frontier``  (P5-§1.4) the 200-image evaluation set, variants ``V0 | V1 | S3-9 | V12 | V2`` over
              the full ``lam`` grid -- the curve that tests the multiplicative-variance corollary.

Judges (all three on every row, so ``p5_judge_validity.csv`` can put them side by side):

``clip_d``    the P4 CLIP directional score (amplitude meter, kept for the comparison)
``er_dim1``   EditReward head 0 (Instruction Following & Semantic Fidelity), pooled at <|Reward|>
``q_follow``  Qwen2.5-VL logP(Yes) - logP(No) for "does the edited image follow the instruction?"

Outputs ``results/p5/<spec>/shards/shard{i}.csv`` (one row per image x variant x lam).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import edit_ops  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.editreward import EditRewardJudge  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now  # noqa: E402
from geoedit.masks import gaussian_blur  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402
from geoedit.vlm_judge import QwenYesNoJudge  # noqa: E402

FIELDS = ["image_id", "category", "variant", "lam", "nfe", "psnr_unedit", "lpips_unedit",
          "clip_d", "clip_target_edit", "delta_norm", "er_dim1", "er_dim2", "q_follow", "q_keep",
          "q_follow_sw", "q_keep_sw", "instruction", "perm_image_id"]

NFE = {"recon": 0, "V0": 2, "V2": 2, "E1a": 4, "E1b": 0, "E1c": 2,
       "V1": 8, "S3_9": 8, "V12": 10,
       "O4_k0": 4, "O4_k1": 4, "O4_k2": 4, "O10_k0": 10, "O10_k1": 10, "O10_k2": 10}
SIGMA_4NFE = 2.0  # L5 from scripts/53_p5_localizer.py: the best single-seed smoother
SDS_K = (0.0, 1.0, 2.0)  # the k in  lam*M*R + k*M^2*R

LAM_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def _diffedit_cont(Rs: torch.Tensor) -> torch.Tensor:
    energy = Rs.norm(dim=1).mean(dim=0)
    return energy / energy.max().clamp_min(1e-12)


def clip_d(feats, src_feat, tar_text, src_text):
    di = feats - src_feat
    dt = (tar_text - src_text).expand_as(di)
    return (di / di.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            * dt / dt.norm(dim=-1, keepdim=True).clamp_min(1e-8)).sum(dim=-1)


def build_perm(dataset) -> Dict[str, str]:
    """Same fixed permutation as P4: another image, never the same category."""
    pool = [i for i in range(len(dataset)) if dataset[i].category not in ("8", "9")]
    rng = np.random.default_rng(20261125)
    by_cat: Dict[str, List[str]] = {}
    for i in pool:
        by_cat.setdefault(dataset[i].category, []).append(dataset[i].image_id)
    perm = {}
    for i in pool:
        sample = dataset[i]
        choices = [x for c, v in by_cat.items() if c != sample.category for x in v]
        perm[sample.image_id] = str(rng.choice(choices))
    return perm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--spec", choices=["validity", "frontier", "ours"], default="validity")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--judge-batch", type=int, default=3)
    parser.add_argument("--swap-mod", type=int, default=5,
                        help="run the swapped-order Qwen question when rank %% swap_mod == 0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validity-lam", type=float, default=1.0,
                        help="lambda for the validity spec (document's pre-registered value is 1.0; "
                             "0.5 is only used for the post-hoc sensitivity run)")
    parser.add_argument("--tag", default=None, help="override the output directory name")
    parser.add_argument("--no-keep", action="store_true",
                        help="skip the preservation question (not needed for the frontiers)")
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    num_shards = int(args.num_shards or base["num_shards"])
    gpus = [int(g) for g in base["gpus"]]
    device = args.device or f"cuda:{gpus[args.shard % len(gpus)]}"

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = [i for i in range(len(dataset)) if dataset[i].category not in ("8", "9")]

    if args.spec in ("frontier", "ours"):
        eval_idx = [pool[i] for i in
                    np.linspace(0, len(pool) - 1, 200).round().astype(int).tolist()]
        picked = [i for k, i in enumerate(eval_idx) if k % num_shards == args.shard]
    else:
        picked = [i for k, i in enumerate(pool) if k % num_shards == args.shard]
    if args.limit:
        picked = picked[: args.limit]
    perm = build_perm(dataset)

    model = SDTurbo(model_root=base["model_root"], device=device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"])
    metrics = PieMetrics(device=device)
    er_judge = EditRewardJudge(device=device)
    qwen = QwenYesNoJudge(device=device, backbone=er_judge)  # shares the 7B backbone

    a_root = ROOT / base["output_root"] / "p3" / "a"
    out_dir = ROOT / base["output_root"] / "p5" / (args.tag or args.spec)
    csv_path = out_dir / "shards" / f"shard{args.shard}.csv"
    done = set()
    if csv_path.exists():
        import csv
        with csv_path.open() as fh:
            done = {row["image_id"] for row in csv.DictReader(fh)}
    print(f"[p5 {args.spec} shard {args.shard}/{num_shards}] {len(picked)} images on {device} "
          f"(done {len(done)})", flush=True)

    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(picked):
        sample = dataset[index]
        if sample.image_id in done:
            continue
        x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt",
                         weights_only=True).float().to(model.device)
        bundle = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
        R9 = torch.stack([bundle[f"t0.9_s{s}"] for s in range(4)], 0).float().to(model.device)
        mask64 = torch.from_numpy(sample.mask64).float().to(model.device)

        entries: List[dict] = [{"variant": "recon", "lam": 0.0, "latent": x_s, "dn": 0.0, "p": ""}]
        if args.spec == "validity":
            R5 = torch.stack([bundle[f"t0.5_s{s}"] for s in range(4)], 0).float().to(model.device)
            M5 = _diffedit_cont(R5)
            other = dataset[perm[sample.image_id]]
            emb_src, emb_tar = model.embed([sample.src_prompt]), model.embed([sample.tar_prompt])
            emb_other = model.embed([other.tar_prompt])
            eps0 = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
            R_wrong = edit_ops.residual(model, x_s, 0.9, emb_src, emb_other, eps0)
            g = torch.Generator(device="cpu").manual_seed(edit_ops.image_seed(sample.image_id, 777))
            xi = torch.randn(R9[0].shape, generator=g, dtype=torch.float32).to(model.device)
            envelope = R9[0].abs() * xi
            base_dn = float(R9[0].norm())
            noise = envelope / envelope.norm().clamp_min(1e-12) * base_dn
            lam_v = float(args.validity_lam)
            deltas = {"V0": (R9[0], ""), "V2": (mask64.unsqueeze(0).unsqueeze(0) * R9[0], ""),
                      "E1a": (R_wrong, perm[sample.image_id]), "E1b": (noise, ""),
                      "E1c": (-R9[0], "")}
            entries += [{"variant": name, "lam": lam_v,
                         "latent": x_s + lam_v * delta, "dn": lam_v * float(delta.norm()),
                         "p": p} for name, (delta, p) in deltas.items()]
        elif args.spec == "frontier":
            M5 = _diffedit_cont(torch.stack([bundle[f"t0.5_s{s}"] for s in range(4)], 0)
                                .float().to(model.device))
            M9 = _diffedit_cont(R9)
            i9 = int(torch.argmin(((R9 * (1.0 - M9).unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=(1, 2, 3))))
            deltas = {"V0": R9[0], "V1": R9.mean(dim=0), "S3_9": R9[i9],
                      "V12": M5.unsqueeze(0).unsqueeze(0) * R9[0],
                      "V2": mask64.unsqueeze(0).unsqueeze(0) * R9[0]}
            for name, delta in deltas.items():
                dn = float(delta.norm())
                for lam in LAM_GRID:
                    entries.append({"variant": name, "lam": lam, "latent": x_s + lam * delta,
                                    "dn": lam * dn, "p": ""})
        else:  # ours: the 4-NFE arm (single-seed smoothed mask) and the 10-NFE arm (4-seed mean)
            e5 = bundle["t0.5_s0"].float().to(model.device).norm(dim=0)
            R5 = torch.stack([bundle[f"t0.5_s{s}"] for s in range(4)], 0).float().to(model.device)
            masks = {"O4": gaussian_blur(e5, SIGMA_4NFE), "O10": R5.norm(dim=1).mean(dim=0)}
            for tag, m in masks.items():
                m = m / m.max().clamp_min(1e-12)
                for k in SDS_K:
                    delta = (m + k * m * m).unsqueeze(0).unsqueeze(0) * R9[0]
                    dn = float(delta.norm())
                    for lam in LAM_GRID:
                        entries.append({"variant": f"{tag}_k{k:g}", "lam": lam,
                                        "latent": x_s + lam * delta, "dn": lam * dn, "p": ""})
        decoded: List[np.ndarray] = []
        for k in range(0, len(entries), args.batch):
            stack = torch.cat([c["latent"] for c in entries[k:k + args.batch]], dim=0).to(model.device)
            decoded.extend(model.decode(stack))
        source = np.asarray(sample.image())
        scored = metrics.evaluate_many(source, decoded, sample.mask512, sample.src_prompt,
                                       sample.tar_prompt, chunk=args.eval_batch)
        cd = clip_d(metrics._image_features(decoded), metrics._image_features([source]),
                    metrics._text_features(sample.tar_prompt),
                    metrics._text_features(sample.src_prompt)).float().cpu()

        pairs = [(sample.image(), Image.fromarray(im), sample.instruction) for im in decoded]
        er = er_judge.score_pairs(pairs, batch_size=args.judge_batch)
        obj = (sample.blended_word.split() or [""])[0]
        q_follow = qwen.score_following(pairs, batch_size=args.judge_batch)
        if args.no_keep:
            q_keep = [float("nan")] * len(pairs)
        else:
            q_keep = qwen.score_keep(pairs, batch_size=args.judge_batch,
                                     objects=[obj] * len(pairs))
        if args.swap_mod and position % args.swap_mod == 0:
            q_follow_sw = qwen.score_following(pairs, batch_size=args.judge_batch, order="swapped")
            q_keep_sw = (qwen.score_keep(pairs, batch_size=args.judge_batch, order="swapped",
                                         objects=[obj] * len(pairs))
                         if not args.no_keep else [float("nan")] * len(pairs))
        else:
            q_follow_sw = [float("nan")] * len(pairs)
            q_keep_sw = [float("nan")] * len(pairs)

        rows = [{"image_id": sample.image_id, "category": sample.category,
                 "variant": e["variant"], "lam": e["lam"], "nfe": NFE.get(e["variant"], 0),
                 "psnr_unedit": s["psnr_unedit"], "lpips_unedit": s["lpips_unedit"],
                 "clip_d": float(c), "clip_target_edit": s["clip_target_edit"],
                 "delta_norm": e["dn"], "er_dim1": float(r[0]), "er_dim2": float(r[1]),
                 "q_follow": float(qf), "q_keep": float(qk), "q_follow_sw": float(qfs),
                 "q_keep_sw": float(qks), "instruction": sample.instruction, "perm_image_id": e["p"]}
                for e, s, c, r, qf, qk, qfs, qks in
                zip(entries, scored, cd, er, q_follow, q_keep, q_follow_sw, q_keep_sw)]
        append_rows(csv_path, rows, FIELDS)
        processed += 1
        if processed % 5 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(picked)} {el / processed:.1f}s/img "
                  f"eta {(len(picked) - position - 1) * el / processed / 60:.1f} min", flush=True)

    print(f"[p5 {args.spec} shard {args.shard}] done={processed} in "
          f"{(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
