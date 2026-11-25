"""Stage C: P2 generation -- every method's latent, per (image, t, seed).

For each image we encode once, then for every ``(t, seed)`` compute

* ``lam``    six scalars from a single residual  (batched)
* ``picard`` K = 1..6, every prefix saved        (sequential, 6 forwards)
* ``avg``    K in {2,4,6} x lambda in {1,2} over independent noises (batched)
* ``prox``   one Tweedie projection of every variant above  (one batched call)

and store everything as one fp16 bundle ``results/p2/latents/{id}_t{t}_s{seed}.pt``.
Diagnostics (norms, contraction ratios, divergence flag, halo rho of each Picard
prefix) go to ``results/p2/shards/shard{i}_diag.csv``.  Resumable: an existing bundle is
skipped, so a killed run continues where it stopped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from geoedit import edit_ops
from geoedit.data import PIEDataset
from geoedit.edit_ops import masked_norms
from geoedit.halo import e_rel, prepare_null, rho, rho_null_prepared
from geoedit.io_utils import append_rows, load_yaml, now, write_manifest
from geoedit.model import SDTurbo

ROOT = Path(__file__).resolve().parents[1]

DIAG_FIELDS = ["image_id", "category", "t", "seed", "k", "nfe_method",
               "delta_in", "delta_out", "delta_ring", "rho_2_6",
               "norm_k", "c_k", "diverged", "res_norm"]


def variant_name(method: str, param: str, mode: str) -> str:
    return f"{method}|{param}|{mode}"


def prox_noise_seed(seed: int, offset: int) -> int:
    return int(seed) + int(offset)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p2-config", default="configs/p2_howfar.yaml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p2 = load_yaml(ROOT / args.p2_config)
    num_shards = args.num_shards or int(base["num_shards"])
    device = args.device or f"cuda:{args.shard % torch.cuda.device_count()}"

    t_list = [float(t) for t in p2["t_list"]]
    seeds_by_t = {float(k): [int(s) for s in v] for k, v in p2["seeds_by_t"].items()}
    lam_list = [float(v) for v in p2["lam_list"]]
    k_list = [int(v) for v in p2["picard_k"]]
    avg_k = [int(v) for v in p2["avg_k"]]
    avg_lam = [float(v) for v in p2["avg_lambda"]]
    t_c = float(p2["t_c"])
    prox_offset = int(p2["prox_seed_offset"])
    div_ratio = float(p2["divergence_ratio"])

    dataset = PIEDataset(pie_root=Path(base["pie_root"]))
    indices = [i for i in range(len(dataset)) if i % num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]

    out_root = ROOT / base["output_root"] / "p2"
    latent_dir = out_root / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_root / "shards"
    diag_csv = shard_dir / f"shard{args.shard}_diag.csv"

    if args.shard == 0:
        write_manifest(out_root / "manifest.json", {
            "stage": "p2_generate",
            "configs": {"base": base, "p2": p2},
            "num_shards": num_shards,
            "grid": {"t_list": t_list, "seeds_by_t": {str(k): v for k, v in seeds_by_t.items()},
                     "lam": lam_list, "picard_k": k_list, "avg_k": avg_k, "avg_lambda": avg_lam},
        })

    model = SDTurbo(model_root=base["model_root"], device=device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"])
    print(f"[p2 gen shard {args.shard}/{num_shards}] {len(indices)} images on {device}", flush=True)

    def bundle_path(image_id: str, t: float, seed: int) -> Path:
        return latent_dir / f"{image_id}_t{t:g}_s{seed}.pt"

    start = time.perf_counter()
    done = skipped = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        expected = [(t, s) for t in t_list for s in seeds_by_t.get(t, [])]
        if all(bundle_path(sample.image_id, t, s).exists() for t, s in expected):
            skipped += 1
            continue
        todo = expected

        image = sample.image()
        x_s = model.encode([image])
        emb_src = model.embed([sample.src_prompt])
        emb_tar = model.embed([sample.tar_prompt])
        prepared = prepare_null(sample.mask64, n_shifts=int(20), gap=6.0,
                                rng=np.random.default_rng(0))
        diag_rows: List[Dict] = []

        for t, seed in todo:
            eps = edit_ops.make_eps(sample.image_id, seed, model.device, model.unet_dtype)
            mu_src = edit_ops.source_mu(model, x_s, t, emb_src, eps)          # 1 NFE
            res = edit_ops.residual_from_mu(model, x_s, t, emb_tar, eps, mu_src)  # 1 NFE

            bundle: Dict[str, torch.Tensor] = {}
            for lam in lam_list:
                bundle[variant_name("lam", f"{lam:g}", "raw")] = edit_ops.lam_edit(x_s, res, lam)

            pic = edit_ops.picard(model, x_s, t, emb_src, emb_tar, eps, k_max=max(k_list),
                                  divergence_ratio=div_ratio, mu_src=mu_src)
            for k in k_list:
                bundle[variant_name("picard", f"{k}", "raw")] = x_s + pic.delta(k)
            bundle[variant_name("picard", "0", "raw")] = x_s

            # avg over independent noises: batch all (K, eps) pairs in one go
            avg_eps = {k: [eps] + [edit_ops.make_eps(sample.image_id, seed * 1000 + 100 + j,
                                                     model.device, model.unet_dtype)
                                  for j in range(1, k)] for k in avg_k}
            flat_eps, flat_index = [], []
            for k in avg_k:
                for j, e in enumerate(avg_eps[k]):
                    flat_eps.append(e)
                    flat_index.append((k, j))
            eps_all = torch.stack(flat_eps)
            xs_all = x_s.expand(len(flat_eps), -1, -1, -1)
            src_all = emb_src.expand(len(flat_eps), -1, -1)
            tar_all = emb_tar.expand(len(flat_eps), -1, -1)
            mu_src_all = edit_ops.source_mu(model, xs_all, t, src_all, eps_all)     # 1 batch
            res_all = edit_ops.residual_from_mu(model, xs_all, t, tar_all, eps_all, mu_src_all)  # 1 batch
            sums = {k: torch.zeros_like(x_s) for k in avg_k}
            for (k, _), r in zip(flat_index, res_all):
                sums[k] = sums[k] + r[None]
            for k in avg_k:
                avg_r = sums[k] / float(k)
                for lam in avg_lam:
                    bundle[variant_name("avg", f"{k}_{lam:g}", "raw")] = edit_ops.lam_edit(x_s, avg_r, lam)

            # prox: one batched Tweedie projection for every variant above
            eps_c = edit_ops.make_eps(sample.image_id, prox_noise_seed(seed, prox_offset),
                                      model.device, model.unet_dtype)
            names = list(bundle.keys())
            stack = torch.cat([bundle[n] for n in names], dim=0)
            proxed = edit_ops.prox(model, stack, t_c, emb_tar.expand(len(names), -1, -1),
                                   eps_c.unsqueeze(0).expand(len(names), -1, -1, -1))
            for n, p in zip(names, proxed):
                bundle[f"{n}|prox".replace("|raw|prox", "|prox")] = p[None]

            torch.save({k: v.detach().to(torch.float16).cpu() for k, v in bundle.items()},
                       bundle_path(sample.image_id, t, seed))

            # diagnostics
            for k in k_list:
                stats = masked_norms(pic.delta(k), torch.from_numpy(sample.mask64).to(model.device),
                                     torch.from_numpy(sample.dist64).to(model.device))
                value, _ = rho(pic.delta(k), sample.mask64, sample.dist64, band=(2, 6), sigma=3.0)
                diag_rows.append({
                    "image_id": sample.image_id, "category": sample.category, "t": t,
                    "seed": seed, "k": k, "nfe_method": 1 + k,
                    "delta_in": stats["delta_in"], "delta_out": stats["delta_out"],
                    "delta_ring": stats["delta_ring"], "rho_2_6": value,
                    "norm_k": pic.norms[k - 1],
                    "c_k": pic.contraction[k - 2] if k >= 2 else float("nan"),
                    "diverged": int(pic.diverged), "res_norm": float(res.norm().item()),
                })
            lam1 = masked_norms(res, torch.from_numpy(sample.mask64).to(model.device), torch.from_numpy(sample.dist64).to(model.device))
            rho1, _ = rho(res, sample.mask64, sample.dist64, band=(2, 6), sigma=3.0)
            diag_rows.append({
                "image_id": sample.image_id, "category": sample.category, "t": t, "seed": seed,
                "k": 0, "nfe_method": 2, "delta_in": lam1["delta_in"],
                "delta_out": lam1["delta_out"], "delta_ring": lam1["delta_ring"],
                "rho_2_6": rho1, "norm_k": float(res.norm().item()), "c_k": float("nan"),
                "diverged": int(pic.diverged), "res_norm": float(res.norm().item()),
            })

        append_rows(diag_csv, diag_rows, DIAG_FIELDS)
        done += 1
        if args.log_every and (position + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - start
            per_image = elapsed / max(1, done + skipped)
            print(f"  {position + 1}/{len(indices)} done={done} skipped={skipped} "
                  f"{per_image:.1f}s/img eta {(len(indices) - position - 1) * per_image / 60:.1f} min",
                  flush=True)

    elapsed = time.perf_counter() - start
    print(f"[p2 gen shard {args.shard}] done={done} skipped={skipped} "
          f"in {elapsed / 60:.1f} min ({now()})", flush=True)
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
