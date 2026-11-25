"""Stage B: compute the residual ``R(x_s)`` for the P1 halo grid and save it as fp16.

Grid: 700 images x t in {0.9, 0.7, 0.5} x seed in {0, 1, 2, 3}.  Each point costs 2 UNet
forwards (source branch + target branch, same noise).  Sharded by image index so the
per-image masks and the VAE encode are reused across all (t, seed).

Outputs
-------
``results/p1/latents/{image_id}_t{t}_s{seed}.pt``   fp16 ``[1, 4, 64, 64]``
``results/p1/shards/{split}/shard{i}_r.csv``        one row per (image, t, seed, split)
and nothing else: ``rho`` / ``rho_null`` / ``e_rel`` are computed from the saved latents in
``11_p1_analyze.py`` (the two sigma settings share the same latent).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from geoedit import edit_ops
from geoedit.data import PIEDataset
from geoedit.io_utils import append_rows, env_snapshot, load_yaml, now, save_latent, write_manifest
from geoedit.model import SDTurbo

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p1-config", default="configs/p1_halo.yaml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p1 = load_yaml(ROOT / args.p1_config)
    num_shards = args.num_shards or int(base["num_shards"])
    device = args.device or f"cuda:{args.shard % torch.cuda.device_count()}"

    t_list = [float(t) for t in p1["t_list"]]
    seeds = [int(s) for s in p1["seeds"]]

    dataset = PIEDataset(pie_root=Path(base["pie_root"]))
    indices = [i for i in range(len(dataset)) if i % num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]

    latent_dir = ROOT / base["output_root"] / "p1" / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = ROOT / base["output_root"] / "p1" / "shards"
    shard_csv = shard_dir / f"shard{args.shard}_rows.csv"

    if args.shard == 0:
        write_manifest(ROOT / base["output_root"] / "p1" / "manifest.json", {
            "stage": "p1_compute",
            "configs": {"base": base, "p1": p1},
            "env": env_snapshot(),
            "num_shards": num_shards,
        })

    model = SDTurbo(model_root=base["model_root"], device=device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"])
    print(f"[shard {args.shard}/{num_shards}] {len(indices)} images on {device}, "
          f"{len(t_list)} t x {len(seeds)} seeds", flush=True)

    start = time.perf_counter()
    done = skipped = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        paths = {t: [latent_dir / f"{sample.image_id}_t{t:g}_s{s}.pt" for s in seeds] for t in t_list}
        if all(p.exists() for t in t_list for p in paths[t]):
            skipped += 1
            continue

        image = sample.image()
        x_s = model.encode([image])  # [1,4,64,64], fp32 scaled latents
        emb_src = model.embed([sample.src_prompt])
        emb_tar = model.embed([sample.tar_prompt])
        eps_batch = torch.stack(
            [edit_ops.make_eps(sample.image_id, s, model.device, model.unet_dtype) for s in seeds]
        )  # [S,4,64,64]
        rows = []
        for t in t_list:
            xs_rep = x_s.expand(len(seeds), -1, -1, -1)
            src_rep = emb_src.expand(len(seeds), -1, -1)
            tar_rep = emb_tar.expand(len(seeds), -1, -1)
            mu_src = edit_ops.source_mu(model, xs_rep, t, src_rep, eps_batch)
            res = edit_ops.residual_from_mu(model, xs_rep, t, tar_rep, eps_batch, mu_src)  # [S,4,64,64]
            for s_index, seed in enumerate(seeds):
                save_latent(paths[t][s_index], res[s_index : s_index + 1])
                row = {
                    "image_id": sample.image_id,
                    "category": sample.category,
                    "t": t,
                    "seed": seed,
                    "nfe": 2,
                    "res_norm": float(res[s_index].norm().item()),
                    "interior_frac": sample.interior_fraction,
                }
                rows.append(row)
        append_rows(shard_csv, rows)
        done += 1
        if args.log_every and (position + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - start
            per_image = elapsed / max(1, done + skipped)
            left = (len(indices) - position - 1) * per_image
            print(f"  {position + 1}/{len(indices)} done={done} skipped={skipped} "
                  f"{per_image:.2f}s/img eta {left / 60:.1f} min", flush=True)

    elapsed = time.perf_counter() - start
    print(f"[shard {args.shard}] done={done} skipped={skipped} in {elapsed / 60:.1f} min "
          f"({now()})", flush=True)
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
