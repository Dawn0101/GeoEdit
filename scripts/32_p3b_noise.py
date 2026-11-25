"""P3b: source-consistent (fixed-point) noise + antithetic noise (doc §4).

For every image and ``t in {0.9, 0.7}``:

1. iterate ``eps^(j+1) = eps_hat_src(a x_s + b eps^(j))`` for ``j = 0 .. 8`` (one source
   forward per step, no backprop) and record ``e_j``, the relative step, and the std /
   excess kurtosis of ``eps^(j)``;
2. build ``R^(J)`` and ``A^(J) = x_hat_tar - x_s`` for ``J in {0, 2, 8}`` and edit at
   ``lambda in {0.5, 1, 1.5, 2}``;
3. ``R(-eps_0)`` for the antithetic comparison.

Everything is decoded and scored with the same batched metrics as P3a, so the rows are
directly comparable with ``results/p3/a``.

Outputs
-------
``results/p3/b/shards/shard{i}.csv``            one row per (image, t, kind, J, lam)
``results/p3/b/shards/shard{i}_diag.csv``       one row per (image, t, j)
``results/p3/b/per_image.csv`` / ``fixed_point_diag.csv``  (merged by 31-style helper)
``results/p3/b/hypotheses.json``                H3e (written by ``31_p3a_analyze.py --b``)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import edit_ops, noise_ops  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now, read_rows, write_manifest  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

FIELDS = ["image_id", "category", "t", "kind", "J", "lam", "nfe",
          "psnr", "lpips", "mse", "ssim", "struct_dist",
          "clip_target_whole", "clip_target_edit", "clip_source_whole"]
DIAG_FIELDS = ["image_id", "category", "t", "j", "e_src", "step", "eps_std", "eps_kurt"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p3-config", default="configs/p3.yaml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / args.p3_config)
    b_cfg = p3["p3b"]
    num_shards = int(args.num_shards or base["num_shards"])
    gpus = [int(g) for g in base["gpus"]]
    device = args.device or f"cuda:{gpus[args.shard % len(gpus)]}"

    t_list = [float(t) for t in b_cfg["t_list"]]
    j_max = int(b_cfg["j_max"])
    j_report = [int(j) for j in b_cfg["J_report"]]
    lam_list = [float(v) for v in b_cfg["lam_list"]]

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    indices = [i for i in range(len(dataset)) if i % num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]

    out = ROOT / base["output_root"] / "p3" / "b"
    shard_dir = out / "shards"
    shard_csv = shard_dir / f"shard{args.shard}.csv"
    diag_csv = shard_dir / f"shard{args.shard}_diag.csv"
    if args.shard == 0:
        write_manifest(out / "manifest.json", {
            "stage": "p3b_noise", "configs": {"base": base, "p3b": b_cfg},
            "num_shards": num_shards, "batch": args.batch})

    model = SDTurbo(model_root=base["model_root"], device=device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"])
    metrics = PieMetrics(device=device)

    done_ids = {row["image_id"] for row in read_rows(shard_csv)}
    print(f"[p3b shard {args.shard}/{num_shards}] {len(indices)} images on {device}"
          f"{f', resuming ({len(done_ids)})' if done_ids else ''}", flush=True)

    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        if sample.image_id in done_ids:
            continue
        x_s = model.encode([sample.image()])
        emb_src = model.embed([sample.src_prompt])
        emb_tar = model.embed([sample.tar_prompt])
        source = np.asarray(sample.image())
        eps0 = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)

        rows: List[Dict] = []
        diag_rows: List[Dict] = []
        latents: List[Dict] = []
        for t in t_list:
            a, b = model.ab(t)
            eps_list, diag = noise_ops.fixed_point_eps(model, x_s, t, emb_src, eps0, j_max=j_max)
            for row in diag:
                diag_rows.append({"image_id": sample.image_id, "category": sample.category,
                                  "t": t, "j": int(row["j"]), "e_src": row["e_src"],
                                  "step": row["step"], "eps_std": row["eps_std"],
                                  "eps_kurt": row["eps_kurt"]})

            # J = 0 is the plain single-draw edit, i.e. V0 of P3a
            for J in j_report:
                eps_j = eps_list[J]
                z = a * x_s + b * eps_j
                mu_src = model.x0_pred(z, t, emb_src)
                mu_tar = model.x0_pred(z, t, emb_tar)
                res = mu_tar - mu_src
                for lam in lam_list:
                    latents.append({"t": t, "kind": "fixed_point", "J": J, "lam": lam,
                                    "latent": x_s + float(lam) * res})
                    latents.append({"t": t, "kind": "sds", "J": J, "lam": lam,
                                    "latent": x_s + float(lam) * (mu_tar - x_s)})
                if J == 0:
                    # §2.1 identity check: A - R - e_src == 0
                    gap = float(((mu_tar - x_s) - res - (mu_src - x_s)).abs().max())
                    diag_rows.append({"image_id": sample.image_id, "category": sample.category,
                                      "t": t, "j": -1, "e_src": gap, "step": float("nan"),
                                      "eps_std": float("nan"), "eps_kurt": float("nan")})

            # antithetic noise (§4.2)
            eps_anti = noise_ops.antithetic(eps0)
            res_anti = edit_ops.residual(model, x_s, t, emb_src, emb_tar, eps_anti)
            for lam in lam_list:
                latents.append({"t": t, "kind": "antithetic", "J": -1, "lam": lam,
                                "latent": x_s + float(lam) * res_anti})

        decoded: List[np.ndarray] = []
        for chunk_start in range(0, len(latents), args.batch):
            stack = torch.cat([c["latent"] for c in latents[chunk_start:chunk_start + args.batch]],
                              dim=0).to(model.device)
            decoded.extend(model.decode(stack))
        scored_all = metrics.evaluate_many(source, decoded, sample.mask512,
                                           sample.src_prompt, sample.tar_prompt,
                                           chunk=args.eval_batch)
        for meta, scored in zip(latents, scored_all):
            rows.append({"image_id": sample.image_id, "category": sample.category,
                         "t": meta["t"], "kind": meta["kind"], "J": meta["J"], "lam": meta["lam"],
                         "nfe": 0,
                         "psnr": scored["psnr_unedit"], "lpips": scored["lpips_unedit"],
                         "mse": scored["mse_unedit"], "ssim": scored["ssim_unedit"],
                         "struct_dist": scored["struct_dist"],
                         "clip_target_whole": scored["clip_target_whole"],
                         "clip_target_edit": scored["clip_target_edit"],
                         "clip_source_whole": scored["clip_source_whole"]})

        append_rows(shard_csv, rows, FIELDS)
        append_rows(diag_csv, diag_rows, DIAG_FIELDS)
        processed += 1
        if args.log_every and processed % args.log_every == 0:
            elapsed = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {elapsed / processed:.1f}s/img "
                  f"eta {(len(indices) - position - 1) * elapsed / processed / 60:.1f} min",
                  flush=True)

    print(f"[p3b shard {args.shard}] done={processed} in "
          f"{(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
