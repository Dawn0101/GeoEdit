"""P4-E1d: judge the variants with the local EditReward model at **matched amplitude**.

CLIP-D turned out to be an amplitude meter (REPORT_p4 §2), so the semantic comparison is
redone with a human-aligned reward model: EditReward (Qwen2.5-VL-7B + two heads, local copy at
/data/lx/Model/Qwen2.5-VL-7B-SFT).  Because CLIP-D cannot serve as the matching anchor any
more, every variant is scaled to the **same edit norm** ``||delta|| = ||R0||`` (the V0 lambda=1
amplitude, i.e. mean 66.0), which is the quantity that CLIP-D was really measuring.

Variants (all at the same amplitude, so the judge compares *direction* only):

``V0``     ``R0`` (single seed)
``V1``     mean of the 4 seeds at t=0.9
``S3_5``   the seed with the smallest off-mask energy (M5 from t=0.5)
``V12``    continuous DiffEdit-style mask (t=0.5) times R0
``V2``     GT mask times R0 (oracle)
``src``    no edit at all (scale reference for the judge)

Outputs ``results/p4/e1d/shards/shard{i}.csv``.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.data import PIEDataset  # noqa: E402
from geoedit.editreward import EditRewardJudge  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

FIELDS = ["image_id", "category", "variant", "lam", "delta_norm", "psnr", "score_dim1",
          "score_dim2", "instruction"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p3-config", default="configs/p3.yaml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--judge-batch", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / "configs/p3.yaml")
    num_shards = int(args.num_shards or base["num_shards"])
    gpus = [int(g) for g in base["gpus"]]
    args.device = args.device or f"cuda:{gpus[args.shard % len(gpus)]}"

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = [i for i in range(len(dataset)) if dataset[i].category not in ("8", "9")]
    eval_idx = [pool[i] for i in np.linspace(0, len(pool) - 1, 200).round().astype(int).tolist()]
    picked = [i for k, i in enumerate(eval_idx) if k % num_shards == args.shard]
    if args.limit:
        picked = picked[: args.limit]

    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"], vae_only=True)
    judge = EditRewardJudge(device=args.device)
    a_root = ROOT / base["output_root"] / "p3" / "a"
    csv_path = ROOT / base["output_root"] / "p4" / "e1d" / "shards" / f"shard{args.shard}.csv"
    done = set()
    if csv_path.exists():
        import csv
        with csv_path.open() as fh:
            done = {row["image_id"] for row in csv.DictReader(fh)}
    print(f"[e1d shard {args.shard}/{num_shards}] {len(picked)} images on {args.device}", flush=True)

    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(picked):
        sample = dataset[index]
        if sample.image_id in done:
            continue
        x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt", weights_only=True).float().to(model.device)
        b = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
        R9 = torch.stack([b[f"t0.9_s{s}"] for s in range(4)], 0).float().to(model.device)
        R5 = torch.stack([b[f"t0.5_s{s}"] for s in range(4)], 0).float().to(model.device)
        mask64 = torch.from_numpy(sample.mask64).float().to(model.device)
        energy5 = R5.norm(dim=1).mean(dim=0)
        M5 = energy5 / energy5.max().clamp_min(1e-12)
        i_star = int(torch.argmin(((R9 * (1.0 - M5).unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=(1, 2, 3))))
        deltas: Dict[str, torch.Tensor] = {
            "V0": R9[0],
            "V1": R9.mean(dim=0, keepdim=True),
            "S3_5": R9[i_star],
            "V12": M5.unsqueeze(0).unsqueeze(0) * R9[0],
            "V2": mask64.unsqueeze(0).unsqueeze(0) * R9[0],
        }
        amp = float(R9[0].norm())

        latents = [{"variant": "src", "lam": 0.0, "latent": x_s, "dn": 0.0}]
        for name, delta in deltas.items():
            lam = amp / max(float(delta.norm()), 1e-9)
            latents.append({"variant": name, "lam": lam, "latent": x_s + lam * delta,
                            "dn": float((lam * delta).norm())})

        decoded: List[np.ndarray] = []
        for k in range(0, len(latents), args.batch):
            stack = torch.cat([c["latent"] for c in latents[k:k + args.batch]], dim=0).to(model.device)
            decoded.extend(model.decode(stack))
        source = np.asarray(sample.image())
        from geoedit.metrics import PieMetrics  # local import: heavy

        if not hasattr(main, "_metrics"):
            main._metrics = PieMetrics(device=args.device)  # type: ignore[attr-defined]
        metrics = main._metrics  # type: ignore[attr-defined]
        scored = metrics.evaluate_many(source, decoded, sample.mask512, sample.src_prompt,
                                       sample.tar_prompt, chunk=4)
        items = [(sample.image(), Image.fromarray(im), sample.tar_prompt) for im in decoded]
        rewards = judge.score_pairs(items, batch_size=args.judge_batch)

        rows = [{"image_id": sample.image_id, "category": sample.category,
                 "variant": m["variant"], "lam": m["lam"], "delta_norm": m["dn"],
                 "psnr": s["psnr_unedit"], "score_dim1": r[0], "score_dim2": r[1],
                 "instruction": sample.tar_prompt}
                for m, s, r in zip(latents, scored, rewards)]
        append_rows(csv_path, rows, FIELDS)
        processed += 1
        if processed % 10 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(picked)} {el / processed:.1f}s/img "
                  f"eta {(len(picked) - position - 1) * el / processed / 60:.1f} min", flush=True)

    print(f"[e1d shard {args.shard}] done={processed} in {(time.perf_counter() - start) / 60:.1f} min ({now()})",
          flush=True)
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
