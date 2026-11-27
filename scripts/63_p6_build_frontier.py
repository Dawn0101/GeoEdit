"""P6 Stage 2: build the frontier --- Ours-4 over (k, gamma, lam) plus the references.

Everything reuses the cached P1 residuals, so a "run" costs only VAE decode + metrics; **no UNet
forward** is needed for any Ours arm (``mu_src`` was cached by ``scripts/60``).

Two modes:

``--mode frontier`` (default)
    Evaluate every ``(k, gamma, lam)`` in ``configs/p6.yaml`` plus ``v0``, ``v12``, ``v2``, ``src``
    and keep scalars only.  This defines ``背景PSNR(lam)`` per variant.

``--mode dump --manifest matched_lam.csv``
    The manifest has ``run,lam,target`` rows (solved on the **mean** PSNR curve, no extrapolation,
    by ``scripts/65``); dump those exact latents to
    ``cache/images/p6/<run>_psnr<target>/<image_id>.png`` so the A/B judge can compare variants at
    matched background PSNR.

ChordEdit is *not* rebuilt here: its 700 images already exist (``cache/images/B1_chord``, official
transport+prox on the same SD-Turbo base) and are staged into ``cache/images/p6/chord`` by
``scripts/63``'s dump mode as well (``--stage-chord``).

Outputs ``results/p6/frontier/{per_image.csv,curves.csv}``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.data import PIEDataset  # noqa: E402
from geoedit.edits import attention_term, l5_mask, ours4_delta  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now, write_json  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

FIELDS = ["image_id", "category", "run", "nfe", "psnr_unedit", "lpips_unedit", "ssim_unedit",
          "struct_dist_unedit", "struct_dist", "clip_target_edit", "clip_target_whole",
          "clip_source_whole", "delta_norm", "lam", "k", "gamma"]
SCORE_KEYS = ("psnr_unedit", "lpips_unedit", "ssim_unedit", "struct_dist_unedit", "struct_dist",
              "clip_target_edit", "clip_target_whole", "clip_source_whole")
CHORD_SRC = ROOT / "cache" / "images" / "B1_chord"


def ours_variants(p6) -> List[Tuple[str, Dict[str, float]]]:
    """Explicit list from ``configs/p6.yaml`` (k=0 does not depend on gamma, so it is listed once)."""
    return [(str(v["name"]), {"k": float(v["k"]), "gamma": float(v["gamma"])})
            for v in p6["ours_variants"]]


def deltas_for(sample, x_s, bundle, mu9, model, p6) -> List[Tuple[str, torch.Tensor, int, float, float]]:
    """(run, delta at lam=1, nfe, k, gamma) for every Ours arm plus the three references."""
    R9 = bundle["t0.9_s0"].float().to(model.device)
    R5 = bundle["t0.5_s0"].float().to(model.device)
    A9 = attention_term(R9, mu9, x_s)
    M = l5_mask(R5, float(p6.get("sigma_4nfe", 2.0)))
    mask = torch.from_numpy(sample.mask64).float().to(model.device)
    out: List[Tuple[str, torch.Tensor, int, float, float]] = []
    for name, prm in ours_variants(p6):
        out.append((name, ours4_delta(R9, A9, M, 1.0, prm["k"], prm["gamma"]), 4, prm["k"], prm["gamma"]))
    out.append(("v0", R9, 2, 0.0, 1.0))
    out.append(("v12", M.unsqueeze(0) * R9, 4, 0.0, 1.0))
    out.append(("v2", mask.unsqueeze(0) * R9, 2, 0.0, 1.0))
    return out


def load_inputs(sample, a_root, mu_root, model):
    x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt", weights_only=True).float().to(model.device)
    x_s = x_s[0] if x_s.dim() == 4 else x_s
    bundle = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
    mu9 = torch.load(mu_root / f"{sample.image_id}.pt",
                     weights_only=True)["t0.9_s0"].float().to(model.device)
    return x_s, bundle, mu9


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p6-config", default="configs/p6.yaml")
    parser.add_argument("--mode", choices=["frontier", "dump", "score-dir"], default="frontier")
    parser.add_argument("--dir", default=None, help="image dir for --mode score-dir")
    parser.add_argument("--tag", default=None, help="run name for --mode score-dir")
    parser.add_argument("--manifest", default=None, help="matched_lam.csv for --mode dump")
    parser.add_argument("--stage-chord", action="store_true", help="copy B1_chord images into p6/")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval200", action="store_true",
                        help="the 200-image eval set (frontier curves); default is all 700")
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p6 = load_yaml(ROOT / args.p6_config)
    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    if args.eval200:
        pool = list(range(len(dataset)))
        indices = [pool[i] for i in np.linspace(0, len(pool) - 1, 200).round().astype(int).tolist()]
        indices = [i for k, i in enumerate(indices) if k % args.num_shards == args.shard]
    else:
        indices = [i for k, i in enumerate(range(len(dataset)))
                   if k % args.num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]
    a_root = ROOT / base["output_root"] / "p3" / "a"
    mu_root = ROOT / base["output_root"] / "p6" / "a" / "musrc"
    out_dir = ROOT / base["output_root"] / "p6" / "frontier"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = ROOT / "cache" / "images" / "p6"

    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision="fp32",
                    text_precision="fp32", vae_only=True)

    if args.stage_chord:
        if not CHORD_SRC.exists():
            raise FileNotFoundError(f"{CHORD_SRC} missing (ChordEdit images from v1)")
        dst = img_root / "chord"
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for image_id in dataset.image_ids:
            src = CHORD_SRC / f"{image_id}.png"
            if src.exists() and not (dst / f"{image_id}.png").exists():
                shutil.copy2(src, dst / f"{image_id}.png")
                n += 1
        print(f"[stage-chord] copied {n} images to {dst}")

    if args.mode == "score-dir":
        from PIL import Image  # noqa: F401
        metrics = PieMetrics(device=args.device)
        src_dir = Path(args.dir)
        run = args.tag or src_dir.name
        csv_path = out_dir / f"runs_{run}.csv"
        done = set()
        if csv_path.exists():
            done = set(pd.read_csv(csv_path, dtype={"image_id": str})["image_id"])
        print(f"[score-dir] {run}: {len(indices)} images from {src_dir}")
        for position, index in enumerate(indices):
            sample = dataset[index]
            path = next((src_dir / f"{sample.image_id}{ext}" for ext in (".png", ".jpg", ".jpeg")
                         if (src_dir / f"{sample.image_id}{ext}").exists()), None)
            if path is None or sample.image_id in done:
                continue
            img = np.asarray(Image.open(path).convert("RGB"))
            s = metrics.evaluate_many(np.asarray(sample.image()), [img], sample.mask512,
                                      sample.src_prompt, sample.tar_prompt, chunk=1)[0]
            row = {"image_id": sample.image_id, "category": sample.category, "run": run,
                   "nfe": float(p6["nfe"]["chord_paper_nfe"]), "delta_norm": float("nan"),
                   "lam": float("nan"), "k": 0.0, "gamma": 1.0}
            row.update({key: s[key] for key in SCORE_KEYS})
            append_rows(csv_path, [row], FIELDS)
            if (position + 1) % 25 == 0:
                print(f"  {position + 1}/{len(indices)}", flush=True)
        print(f"[score-dir] done ({now()})")
        return 0

    if args.mode == "dump":
        if not args.manifest:
            raise SystemExit("--mode dump needs --manifest matched_lam.csv")
        spec = pd.read_csv(args.manifest)
        pairs = [(str(r["run"]), float(r["lam"]), str(r["target"])) for _, r in spec.iterrows()]
        print(f"[dump] {len(pairs)} (run,lam,target) rows, {len(indices)} images on {args.device}")
        for position, index in enumerate(indices):
            sample = dataset[index]
            x_s, bundle, mu9 = load_inputs(sample, a_root, mu_root, model)
            table = {name: delta for name, delta, *_ in
                     deltas_for(sample, x_s, bundle, mu9, model, p6)}
            for run, lam, target in pairs:
                if run not in table:
                    continue
                out_dir_run = img_root / f"{run}_psnr{target}"
                out_dir_run.mkdir(parents=True, exist_ok=True)
                path = out_dir_run / f"{sample.image_id}.png"
                if path.exists():
                    continue
                latent = x_s + float(lam) * table[run]
                arr = model.decode(latent.unsqueeze(0))[0]
                from PIL import Image
                Image.fromarray(arr).save(path)
            if (position + 1) % 50 == 0:
                print(f"  {position + 1}/{len(indices)}", flush=True)
        print(f"[dump] done ({now()})")
        return 0

    # ---------------- metrics frontier ----------------
    metrics = PieMetrics(device=args.device)
    lams = [float(x) for x in p6["lam_list"]]
    csv_path = out_dir / "per_image.csv"
    done = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path, dtype={"image_id": str}, on_bad_lines="skip")
        done = set(zip(prev["image_id"], prev["run"], prev["lam"].astype(float)))
    print(f"[frontier shard {args.shard}/{args.num_shards}] {len(indices)} images on {args.device}",
          flush=True)

    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        x_s, bundle, mu9 = load_inputs(sample, a_root, mu_root, model)
        entries: List[dict] = [{"run": "src", "latent": x_s, "lam": 0.0, "dn": 0.0,
                                "nfe": 0, "k": 0.0, "gamma": 1.0}]
        for name, delta, nfe, k, gamma in deltas_for(sample, x_s, bundle, mu9, model, p6):
            dn = float(delta.norm())
            for lam in lams:
                entries.append({"run": name, "latent": x_s + lam * delta, "lam": lam,
                                "dn": lam * dn, "nfe": nfe, "k": k, "gamma": gamma})
        todo = [e for e in entries if (sample.image_id, e["run"], e["lam"]) not in done]
        if not todo:
            continue
        decoded = [model.decode(torch.stack([e["latent"] for e in todo[i:i + 8]]))
                   for i in range(0, len(todo), 8)]
        decoded = [im for block in decoded for im in block]
        source = np.asarray(sample.image())
        scored = metrics.evaluate_many(source, decoded, sample.mask512, sample.src_prompt,
                                       sample.tar_prompt, chunk=8)
        rows = []
        for e, s in zip(todo, scored):
            row = {"image_id": sample.image_id, "category": sample.category, "run": e["run"],
                   "nfe": e["nfe"], "delta_norm": e["dn"], "lam": e["lam"], "k": e["k"],
                   "gamma": e["gamma"]}
            row.update({key: s[key] for key in SCORE_KEYS})
            rows.append(row)
        append_rows(csv_path, rows, FIELDS)
        processed += 1
        if processed % 20 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / processed:.2f}s/img "
                  f"eta {(len(indices) - position - 1) * el / processed / 60:.1f} min", flush=True)

    print(f"[frontier shard {args.shard}] done={processed} in "
          f"{(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    write_json(out_dir / f"p6_frontier_shard{args.shard}.json",
               {"created": now(), "shard": args.shard, "num_shards": args.num_shards,
                "n_images": processed, "lams": lams})
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
