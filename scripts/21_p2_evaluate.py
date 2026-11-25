"""Stage D: decode every P2 latent, run the official PIE-Bench metrics, write per_image.csv.

For each ``(image, t, seed)`` bundle the script walks all variants (``lam``, ``picard``,
``avg``, plus ``recon`` = the ``picard|0`` entry), decodes them in batches, and scores each
one with the official PnPInversion calculators.  ``recon`` (VAE round trip) is the ceiling
reference for PSNR.

Rows also carry the latent diagnostics of that variant's displacement
``x - x_s``: ``delta_in`` / ``delta_out`` / ``delta_ring`` and the halo statistic
``rho_2_6`` (no null here -- P1 owns the null comparison).

Outputs
-------
``results/p2/per_image.csv``  (header written by the first shard to finish)
``results/p2/shards/shard{i}_eval.csv``
``results/p2/png_subset/``    fixed 50 images, t=0.9 seed=0, all raw variants
``results/p2/grid_{id}.png``  6 comparison grids
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from geoedit.data import PIEDataset
from geoedit.edit_ops import masked_norms
from geoedit.halo import rho
from geoedit.io_utils import append_rows, load_yaml, now, read_rows
from geoedit.metrics import PieMetrics
from geoedit.model import SDTurbo

ROOT = Path(__file__).resolve().parents[1]

diag_lookup: Dict[tuple, str] = {}

FIELDS = ["image_id", "category", "t", "seed", "method", "param", "prox", "nfe",
          "struct_dist", "psnr", "lpips", "mse", "ssim", "clip_whole", "clip_edited",
          "delta_in", "delta_out", "delta_ring", "rho_2_6", "diverged", "diverged_flag"]


def parse_variant(name: str) -> tuple:
    """``lam|1|raw`` -> prox 0; ``lam|1|prox`` -> prox 1 (generation rewrites the suffix)."""
    parts = name.split("|")
    method, param = parts[0], parts[1]
    if len(parts) >= 4:
        mode, prox = parts[2], int(parts[3] == "prox")
    else:
        mode = parts[2]
        prox = int(mode == "prox")
    return method, param, mode, prox


def variant_nfe(method: str, param: str) -> int:
    if method == "recon":
        return 0
    if method == "lam":
        return 2
    if method == "picard":
        return 1 + int(param)
    if method == "avg":
        k = int(param.split("_")[0])
        return 2 * k
    raise ValueError(method)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p2-config", default="configs/p2_howfar.yaml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--save-png", action="store_true", default=True)
    parser.add_argument("--no-save-png", dest="save_png", action="store_false")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p2 = load_yaml(ROOT / args.p2_config)
    num_shards = args.num_shards or int(base["num_shards"])
    device = args.device or f"cuda:{args.shard % torch.cuda.device_count()}"

    dataset = PIEDataset(pie_root=Path(base["pie_root"]))
    out_root = ROOT / base["output_root"] / "p2"
    latent_root = out_root / "latents"
    shard_dir = out_root / "shards"
    shard_csv = shard_dir / f"shard{args.shard}_eval.csv"
    png_dir = out_root / "png_subset"
    if args.save_png:
        png_dir.mkdir(parents=True, exist_ok=True)

    # fixed 50-image subset (same even stride as the sanity audit) for the contact sheets
    subset_n = int(p2.get("png_subset_n", 50))
    subset_indices = set(np.linspace(0, len(dataset) - 1, subset_n).round().astype(int).tolist())

    # divergence flags come from the generation diagnostics
    for path in sorted(shard_dir.glob("*_diag.csv")):
        for row in read_rows(path):
            if row.get("diverged") in ("0", "1"):
                diag_lookup[(row["image_id"], float(row["t"]), int(row["seed"]),
                             int(row["k"]))] = row["diverged"]

    model = SDTurbo(model_root=base["model_root"], device=device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"])
    metrics = PieMetrics(device=device)

    indices = [i for i in range(len(dataset)) if i % num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]
    print(f"[p2 eval shard {args.shard}/{num_shards}] {len(indices)} images on {device}", flush=True)

    done_ids = set()
    if shard_csv.exists():
        for row in read_rows(shard_csv):
            done_ids.add(row["image_id"])
        print(f"[resume] {len(done_ids)} images already evaluated in {shard_csv.name}", flush=True)

    start = time.perf_counter()
    done = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        if sample.image_id in done_ids:
            done += 1
            continue
        bundles = []
        for t in [float(v) for v in p2["t_list"]]:
            for seed in [int(v) for v in p2["seeds_by_t"][t]]:
                path = latent_root / f"{sample.image_id}_t{t:g}_s{seed}.pt"
                if path.exists():
                    bundles.append((t, seed, torch.load(path, map_location="cpu",
                                                        weights_only=True)))
        if not bundles:
            continue

        source = np.asarray(sample.image())
        rows: List[Dict] = []
        for t, seed, bundle in bundles:
            x_s = bundle["picard|0|raw"].float().to(device)
            names = sorted(bundle.keys())
            # decode in batches of `args.batch`
            decoded: Dict[str, np.ndarray] = {}
            for start_i in range(0, len(names), args.batch):
                chunk = names[start_i:start_i + args.batch]
                latents = torch.cat([bundle[n].float() for n in chunk], dim=0).to(device)
                for name, image in zip(chunk, model.decode(latents)):
                    decoded[name] = image

            for name in names:
                method, param, mode, prox = parse_variant(name)
                if method == "picard" and param == "0":
                    method = "recon"
                output = decoded[name]
                scored = metrics.evaluate(source, output, sample.mask512,
                                          sample.src_prompt, sample.tar_prompt, lean=True)
                displacement = bundle[name].float().to(device) - x_s
                stats = masked_norms(displacement, torch.from_numpy(sample.mask64).to(device),
                                     torch.from_numpy(sample.dist64).to(device))
                halo, _ = rho(displacement, sample.mask64, sample.dist64, band=(2, 6), sigma=3.0)
                row = {
                    "image_id": sample.image_id, "category": sample.category, "t": t,
                    "seed": seed, "method": method, "param": param, "prox": prox,
                    "nfe": variant_nfe(method, param) + (1 if prox else 0),
                    "struct_dist": scored["struct_dist"],
                    "psnr": scored["psnr_unedit"], "lpips": scored["lpips_unedit"],
                    "mse": scored["mse_unedit"], "ssim": scored["ssim_unedit"],
                    "clip_whole": scored["clip_whole"], "clip_edited": scored["clip_edited"],
                    "delta_in": stats["delta_in"], "delta_out": stats["delta_out"],
                    "delta_ring": stats["delta_ring"], "rho_2_6": halo,
                    "diverged": "", "diverged_flag": "",
                }
                rows.append(row)

                if (args.save_png and index in subset_indices and t == 0.9 and seed == 0
                        and not prox):
                    name_png = f"{sample.image_id}_{method}_{param}_raw.jpg"
                    Image.fromarray(output).save(png_dir / name_png, quality=92)

        # attach the divergence flag of the matching Picard run
        for row in rows:
            if row["method"] == "picard":
                key = (row["image_id"], row["t"], row["seed"], int(row["param"]))
                row["diverged_flag"] = diag_lookup.get(key, "")
        append_rows(shard_csv, rows, FIELDS)
        done += 1
        if args.log_every and (position + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - start
            per_image = elapsed / max(1, done)
            print(f"  {position + 1}/{len(indices)} {per_image:.1f}s/img "
                  f"eta {(len(indices) - position - 1) * per_image / 60:.1f} min", flush=True)

    elapsed = time.perf_counter() - start
    print(f"[p2 eval shard {args.shard}] done={done} in {elapsed / 60:.1f} min ({now()})", flush=True)
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
