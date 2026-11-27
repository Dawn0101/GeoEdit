"""P7: Ours-5 -- the 5-forward arm that matches ChordEdit's measured NFE.

What P6 lost to ChordEdit and what this script adds (per the P7 spec):

1. **dual-scale fusion** (0 NFE): the low-noise residual ``R(t_l=0.5)`` was already computed for
   the mask, but the *edit* only used ``R(t_h=0.9)``.  ``u = u_h + w * (||u_h||/||u_l||) * u_l``
   puts the two scales on equal footing in norm, so the fusion is about direction, not magnitude.
2. **direction gating** (0 NFE): ``gamma_dir = max(0, cos(u_h, u_l))`` on the final field
   (PreciseEdit eq. 9).
3. **core-confined SDS + energy normalisation** (0 NFE): the attraction enters as
   ``k * M**gamma * A_h`` (``gamma in {2,3}``) and the whole displacement is rescaled to
   ``lam * ||R_h||``.  Without the rescale the step length grows with ``||A|| = 1.77 ||R||`` and
   the image is destroyed (P6: PSNR 30 -> 19 dB).
4. **masked prox** (+1 NFE = 5): ``x = (1-M) * x_s + M * prox_{t_c=0.3}(y_edit)`` where the prox is
   the same target-branch ``x0`` prediction ChordEdit does as its 5th forward.  Ablations:
   ``prox=none`` (4 NFE) and ``prox=full`` (M = 1, i.e. ChordEdit-style global cleanup).

Modes mirror ``scripts/63``: ``--mode frontier`` writes scalars for every (variant, lam);
``--mode dump --manifest`` writes the matched-PSNR images for the A/B judge.
"""

from __future__ import annotations

import argparse
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

from geoedit import edit_ops  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.edits import (attention_term, direction_gate, fuse_scales, l5_mask,  # noqa: E402
                           masked_prox, ours5_delta, prox_latent)
from geoedit.io_utils import append_rows, load_yaml, now, write_json  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

FIELDS = ["image_id", "category", "run", "nfe", "psnr_unedit", "lpips_unedit", "ssim_unedit",
          "struct_dist_unedit", "struct_dist", "clip_target_edit", "clip_target_whole",
          "clip_source_whole", "delta_norm", "raw_norm", "lam", "gate", "k", "gamma", "fusion_w"]
SCORE_KEYS = ("psnr_unedit", "lpips_unedit", "ssim_unedit", "struct_dist_unedit", "struct_dist",
              "clip_target_edit", "clip_target_whole", "clip_source_whole")


def load_inputs(sample, a_root, mu_root, model):
    x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt", weights_only=True).float().to(model.device)
    x_s = x_s[0] if x_s.dim() == 4 else x_s
    bundle = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
    mu = torch.load(mu_root / f"{sample.image_id}.pt", weights_only=True)
    R9 = bundle["t0.9_s0"].float().to(model.device)
    R5 = bundle["t0.5_s0"].float().to(model.device)
    mu9 = mu["t0.9_s0"].float().to(model.device)
    return x_s, R9, R5, mu9


def build_variant(variant, x_s, R9, R5, mu9, model, sample, lam, p7, eps, emb_tar):
    """Return (latent, info) for one (variant, lam).  Costs 1 UNet forward iff the variant uses prox."""
    A9 = attention_term(R9, mu9, x_s)
    M = l5_mask(R5, float(p7.get("sigma_4nfe", 2.0)))
    u = fuse_scales(R9, R5, float(variant["fusion_w"]))
    gate = direction_gate(R9, R5) if variant["gate"] else 1.0
    u = gate * u
    k = float(p7["k"])
    delta_raw = ours5_delta(u, A9, M, lam, k, float(variant["gamma"]), reference=None)
    delta = ours5_delta(u, A9, M, lam, k, float(variant["gamma"]), reference=R9) \
        if p7.get("energy_normalise", True) else delta_raw
    y = x_s + delta
    if variant["prox"] == "none":
        return y, {"delta_norm": float(delta.norm()), "raw_norm": float(delta_raw.norm()),
                   "gate": gate}
    prox = prox_latent(model, y, float(p7["prox_t"]), emb_tar, eps)
    if variant["prox"] == "full":
        out = prox
    else:
        # sharpen the composite mask: the soft L5 tails would otherwise blend the prox
        # (which carries e_src) into the background and undo the preservation
        M_c = M ** float(p7.get("prox_mask_gamma", 2.0))
        out = masked_prox(x_s, y, prox, M_c)
    return out, {"delta_norm": float(delta.norm()), "raw_norm": float(delta_raw.norm()), "gate": gate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p7-config", default="configs/p7.yaml")
    parser.add_argument("--mode", choices=["frontier", "dump"], default="frontier")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", default=None, help="comma separated subset of variant names")
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p7 = load_yaml(ROOT / args.p7_config)
    lams = [float(x) for x in p7["lam_list"]]
    variants = p7["variants"]
    if args.variants:
        keep = set(args.variants.split(","))
        variants = [v for v in variants if v["name"] in keep]

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = list(range(len(dataset)))
    eval_idx = [pool[i] for i in np.linspace(0, len(pool) - 1, int(p7.get("n_images", 200)))
                .round().astype(int).tolist()]
    indices = [i for k, i in enumerate(eval_idx) if k % args.num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]

    a_root = ROOT / base["output_root"] / "p3" / "a"
    mu_root = ROOT / base["output_root"] / "p6" / "a" / "musrc"
    out_dir = ROOT / base["output_root"] / "p7"
    (out_dir / "frontier").mkdir(parents=True, exist_ok=True)
    img_root = ROOT / "cache" / "images" / "p7"

    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision="fp32",
                    text_precision="fp32")
    print(f"[p7 {args.mode} shard {args.shard}/{args.num_shards}] {len(indices)} images, "
          f"{len(variants)} variants x {len(lams)} lam on {args.device}", flush=True)

    if args.mode == "dump":
        spec = pd.read_csv(args.manifest)
        rows = [(str(r["run"]), float(r["lam"]), str(r["target"])) for _, r in spec.iterrows()]
        for position, index in enumerate(indices):
            sample = dataset[index]
            x_s, R9, R5, mu9 = load_inputs(sample, a_root, mu_root, model)
            eps = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
            emb_tar = model.embed([sample.tar_prompt])
            for run, lam, target in rows:
                variant = next((v for v in variants if v["name"] == run), None)
                if variant is None:
                    continue
                d = img_root / f"{run}_psnr{target}"
                d.mkdir(parents=True, exist_ok=True)
                path = d / f"{sample.image_id}.png"
                if path.exists():
                    continue
                latent, _ = build_variant(variant, x_s, R9, R5, mu9, model, sample, lam, p7,
                                          eps, emb_tar)
                from PIL import Image
                Image.fromarray(model.decode(latent.unsqueeze(0))[0]).save(path)
            if (position + 1) % 25 == 0:
                print(f"  {position + 1}/{len(indices)}", flush=True)
        print(f"[p7 dump] done ({now()})")
        return 0

    metrics = PieMetrics(device=args.device)
    csv_path = out_dir / "frontier" / (f"per_image_shard{args.shard}.csv" if args.num_shards > 1
                                       else "per_image.csv")
    done = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path, dtype={"image_id": str}, on_bad_lines="skip")
        done = set(zip(prev["image_id"], prev["run"], prev["lam"].astype(float)))
    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        x_s, R9, R5, mu9 = load_inputs(sample, a_root, mu_root, model)
        eps = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
        emb_tar = model.embed([sample.tar_prompt])
        entries: List[dict] = [{"run": "src", "latent": x_s, "lam": 0.0, "nfe": 0, "gate": 1.0,
                                "k": 0.0, "gamma": 1.0, "fusion_w": 0.0,
                                "delta_norm": 0.0, "raw_norm": 0.0}]
        for variant in variants:
            for lam in lams:
                lat, info = build_variant(variant, x_s, R9, R5, mu9, model, sample, lam, p7,
                                          eps, emb_tar)
                entries.append({"run": variant["name"], "latent": lat, "lam": lam,
                                "nfe": variant["nfe"], "k": float(p7["k"]),
                                "gamma": float(variant["gamma"]), "fusion_w": variant["fusion_w"],
                                "delta_norm": info["delta_norm"], "raw_norm": info["raw_norm"],
                                "gate": info["gate"]})
        # a plain V0 reference for the figure
        for lam in lams:
            entries.append({"run": "v0", "latent": x_s + lam * R9, "lam": lam, "nfe": 2,
                            "gate": 1.0, "k": 0.0, "gamma": 1.0, "fusion_w": 0.0,
                            "delta_norm": lam * float(R9.norm()), "raw_norm": lam * float(R9.norm())})
        todo = [e for e in entries if (sample.image_id, e["run"], e["lam"]) not in done]
        if not todo:
            continue
        decoded = []
        for i in range(0, len(todo), 6):
            stack = torch.stack([e["latent"] for e in todo[i:i + 6]])
            decoded.extend(model.decode(stack))
        source = np.asarray(sample.image())
        scored = metrics.evaluate_many(source, decoded, sample.mask512, sample.src_prompt,
                                       sample.tar_prompt, chunk=8)
        rows = []
        for e, s in zip(todo, scored):
            row = {"image_id": sample.image_id, "category": sample.category, "run": e["run"],
                   "nfe": e["nfe"], "delta_norm": e["delta_norm"], "raw_norm": e["raw_norm"],
                   "lam": e["lam"], "gate": e["gate"], "k": e["k"], "gamma": e["gamma"],
                   "fusion_w": e["fusion_w"]}
            row.update({key: s[key] for key in SCORE_KEYS})
            rows.append(row)
        append_rows(csv_path, rows, FIELDS)
        processed += 1
        if processed % 10 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / processed:.1f}s/img eta "
                  f"{(len(indices) - position - 1) * el / processed / 60:.1f} min", flush=True)
    print(f"[p7 frontier shard {args.shard}] done={processed} in "
          f"{(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    write_json(out_dir / "frontier" / f"p7_shard{args.shard}.json",
               {"created": now(), "shard": args.shard, "num_shards": args.num_shards,
                "n_images": processed, "lams": lams})
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
