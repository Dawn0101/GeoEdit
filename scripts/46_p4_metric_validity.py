"""P4 §5 (was P3e-E1): is CLIP-D a real semantic metric, or just an amplitude meter?

Three controls, all at t=0.9 with the P1 residuals (lambda in {0.5, 1, 2}):

``E1a``  wrong prompt: the residual is computed with **another image's** target prompt
         (random permutation, never the same category), while the evaluation still uses this
         image's correct prompt pair.  Needs 2 extra UNet forwards per image.
``E1b``  pure noise with the same envelope and the same norm as ``lam * R0``.
``E1c``  reversed: ``-lam * R0``.

The reference is V0 (``lam * R0``).  The judgement (§5) compares the **CLIP-D per unit
amplitude** of E1a/E1b against V0; E1c must be negative.

Outputs ``results/p4/metric_validity/shards/shard{i}.csv``; the permutation is printed and
saved so the control is auditable.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import edit_ops  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

FIELDS = ["image_id", "category", "variant", "lam", "psnr", "clip_target_edit", "clip_d",
          "delta_norm", "perm_image_id"]
LAMS = [0.5, 1.0, 2.0]


def clip_d(feats, src_feat, tar_text, src_text):
    di = feats - src_feat
    dt = (tar_text - src_text).expand_as(di)
    return (di / di.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            * dt / dt.norm(dim=-1, keepdim=True).clamp_min(1e-8)).sum(dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    num_shards = int(args.num_shards or base["num_shards"])
    gpus = [int(g) for g in base["gpus"]]
    device = args.device or f"cuda:{gpus[args.shard % len(gpus)]}"

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = [i for i in range(len(dataset)) if dataset[i].category not in ("8", "9")]
    indices = [i for k, i in enumerate(pool) if k % num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]
    ids = [dataset[i].image_id for i in pool]

    # fixed permutation: another image, never the same category
    rng = np.random.default_rng(20261125)
    perm: Dict[str, str] = {}
    by_cat: Dict[str, List[str]] = {}
    for i in pool:
        by_cat.setdefault(dataset[i].category, []).append(dataset[i].image_id)
    for i in pool:
        sample = dataset[i]
        choices = [x for c, v in by_cat.items() if c != sample.category for x in v]
        perm[sample.image_id] = str(rng.choice(choices))

    model = SDTurbo(model_root=base["model_root"], device=device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"])
    metrics = PieMetrics(device=device)
    a_root = ROOT / base["output_root"] / "p3" / "a"
    csv_path = ROOT / base["output_root"] / "p4" / "metric_validity" / "shards" / f"shard{args.shard}.csv"
    done = set()
    if csv_path.exists():
        import csv
        with csv_path.open() as fh:
            done = {row["image_id"] for row in csv.DictReader(fh)}
    print(f"[e1 shard {args.shard}/{num_shards}] {len(indices)} images on {device}", flush=True)

    if args.shard == 0:
        (ROOT / base["output_root"] / "p4" / "metric_validity").mkdir(parents=True, exist_ok=True)
        (ROOT / base["output_root"] / "p4" / "metric_validity" / "permutation.json").write_text(
            json.dumps(perm, indent=1))

    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        if sample.image_id in done:
            continue
        x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt", weights_only=True).float().to(model.device)
        bundle = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
        R0 = bundle["t0.9_s0"].float().to(model.device)
        emb_src, emb_tar = model.embed([sample.src_prompt]), model.embed([sample.tar_prompt])

        # E1a: residual against another image's target prompt, same eps and same t
        other = dataset[perm[sample.image_id]]
        emb_other = model.embed([other.tar_prompt])
        eps0 = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
        R_wrong = edit_ops.residual(model, x_s, 0.9, emb_src, emb_other, eps0)

        g = torch.Generator(device="cpu").manual_seed(edit_ops.image_seed(sample.image_id, 777))
        xi = torch.randn(R0.shape, generator=g, dtype=torch.float32).to(model.device)
        envelope = R0.abs() * xi

        latents = [{"variant": "recon", "lam": 0.0, "latent": x_s, "dn": 0.0, "perm": ""}]
        for lam in LAMS:
            base_dn = float(R0.norm())
            latents += [
                {"variant": "V0", "lam": lam, "latent": x_s + lam * R0, "dn": lam * base_dn, "perm": ""},
                {"variant": "E1a", "lam": lam, "latent": x_s + lam * R_wrong, "dn": lam * float(R_wrong.norm()),
                 "perm": perm[sample.image_id]},
                {"variant": "E1c", "lam": lam, "latent": x_s - lam * R0, "dn": lam * base_dn, "perm": ""},
            ]
            noise = envelope / envelope.norm().clamp_min(1e-12) * (lam * base_dn)
            latents.append({"variant": "E1b", "lam": lam, "latent": x_s + noise,
                            "dn": float(noise.norm()), "perm": ""})

        decoded: List[np.ndarray] = []
        for k in range(0, len(latents), args.batch):
            stack = torch.cat([c["latent"] for c in latents[k:k + args.batch]], dim=0).to(model.device)
            decoded.extend(model.decode(stack))
        source = np.asarray(sample.image())
        scored = metrics.evaluate_many(source, decoded, sample.mask512, sample.src_prompt,
                                       sample.tar_prompt, chunk=args.eval_batch)
        cd = clip_d(metrics._image_features(decoded), metrics._image_features([source]),
                    metrics._text_features(sample.tar_prompt),
                    metrics._text_features(sample.src_prompt)).float().cpu()

        rows = [{"image_id": sample.image_id, "category": sample.category,
                 "variant": m["variant"], "lam": m["lam"],
                 "psnr": s["psnr_unedit"], "clip_target_edit": s["clip_target_edit"],
                 "clip_d": float(c), "delta_norm": m["dn"], "perm_image_id": m["perm"]}
                for m, s, c in zip(latents, scored, cd)]
        append_rows(csv_path, rows, FIELDS)
        processed += 1
        if processed % 20 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / processed:.1f}s/img", flush=True)

    print(f"[e1 shard {args.shard}] done={processed} in {(time.perf_counter() - start) / 60:.1f} min ({now()})",
          flush=True)
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
