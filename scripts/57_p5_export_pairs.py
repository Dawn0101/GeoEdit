"""P5-§1.3 fallback: the 50 side-by-side comparisons the PI has to look at.

The pre-registration says: if no judge passes the four validity checks, stop before §3 and hand the
PI a human-checkable set.  This script exports, for 50 images drawn with a fixed seed from the
200-image evaluation set, a four-panel strip

    source | V0 (lam*R0) | V12 (t=0.5 mask * R0) | V2 (GT mask * R0)

all at ``lam=1``, together with the instruction, the background PSNR and the scores of all three
judges, so the PI can decide by eye whether "V0 follows the instruction better than the source"
and whether V12/V2 look better than V0.

Runs on CPU (the VAE decode is the only compute) so it does not contend with the sweeps.

Outputs ``results/p5/pairs/<image_id>.png`` + ``p5_pairs_manifest.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

SEED = 20261126


def _font(size: int = 16):
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = [i for i in range(len(dataset)) if dataset[i].category not in ("8", "9")]
    eval_idx = [pool[i] for i in np.linspace(0, len(pool) - 1, 200).round().astype(int).tolist()]
    rng = np.random.default_rng(SEED)
    picked = sorted(rng.choice(len(eval_idx), size=min(args.n, len(eval_idx)), replace=False).tolist())
    picked = [eval_idx[k] for k in picked]
    print(f"[pairs] {len(picked)} images from the 200-image eval set (seed {SEED}) on {args.device}")

    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision="fp32",
                    text_precision="fp32", vae_only=True)
    a_root = ROOT / base["output_root"] / "p3" / "a"
    out_dir = ROOT / base["output_root"] / "p5" / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(ROOT / base["output_root"] / "p5" / "validity" / "p5_judge_validity.csv",
                      dtype={"variant": str}) if (ROOT / base["output_root"] / "p5" / "validity" /
                                                  "p5_judge_validity.csv").exists() else None
    val_shards = sorted((ROOT / base["output_root"] / "p5" / "validity" / "shards").glob("shard*.csv"))
    val_frame = pd.concat([pd.read_csv(p, dtype={"image_id": str, "category": str})
                           for p in val_shards], ignore_index=True) if val_shards else None
    if val_frame is not None:
        val_frame = val_frame.drop_duplicates(subset=["image_id", "variant"])

    font, small = _font(18), _font(14)
    rows = []
    for k, index in enumerate(picked):
        sample = dataset[index]
        x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt", weights_only=True).float()
        bundle = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
        R9 = torch.stack([bundle[f"t0.9_s{s}"] for s in range(4)], 0).float()
        R5 = torch.stack([bundle[f"t0.5_s{s}"] for s in range(4)], 0).float()
        M5 = R5.norm(dim=1).mean(dim=0)
        M5 = M5 / M5.max().clamp_min(1e-12)
        mask = torch.from_numpy(sample.mask64).float()
        variants = {
            "source": x_s,
            "V0": x_s + R9[0],
            "V12": x_s + M5.unsqueeze(0).unsqueeze(0) * R9[0],
            "V2": x_s + mask.unsqueeze(0).unsqueeze(0) * R9[0],
        }
        decoded = {}
        for name, latent in variants.items():
            decoded[name] = model.decode(latent.unsqueeze(0) if latent.dim() == 3 else latent)[0]

        strip = Image.new("RGB", (512 * 4, 512 + 96), (255, 255, 255))
        for j, name in enumerate(["source", "V0", "V12", "V2"]):
            strip.paste(Image.fromarray(decoded[name]), (512 * j, 0))
        draw = ImageDraw.Draw(strip)
        for j, name in enumerate(["source", "V0", "V12", "V2"]):
            draw.text((512 * j + 8, 512 + 4), f"{name} (lam=1)", fill=(0, 0, 0), font=font)
            sub = ""
            if val_frame is not None and name != "source":
                blk = val_frame[(val_frame.image_id == sample.image_id) & (val_frame.variant == name)]
                if len(blk):
                    r = blk.iloc[0]
                    sub = (f"PSNR {r.psnr_unedit:.1f}  ER {r.er_dim1:+.2f}  "
                           f"Qwen {r.q_follow:+.2f}")
            draw.text((512 * j + 8, 512 + 28), sub, fill=(60, 60, 60), font=small)
        draw.text((8, 512 + 48), f"{sample.image_id} (cat {sample.category})  "
                                f"instruction: {sample.instruction}", fill=(0, 0, 0), font=small)
        strip.save(out_dir / f"{sample.image_id}.png")
        rows.append({"image_id": sample.image_id, "category": sample.category,
                     "instruction": sample.instruction,
                     "src_prompt": sample.src_prompt, "tar_prompt": sample.tar_prompt,
                     "blended_word": sample.blended_word,
                     "path": str((out_dir / f"{sample.image_id}.png").relative_to(ROOT))})
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(picked)}", flush=True)

    pd.DataFrame(rows).to_csv(out_dir / "p5_pairs_manifest.csv", index=False)
    write_json(out_dir / "p5_pairs_meta.json", {
        "created": now(), "seed": SEED, "n": len(rows), "source": "200-image eval set (200 sampled "
        "uniformly from the 540-image pool of categories 0-7)", "lam": 1.0,
        "reason": "§1.3: no judge passed the four pre-registered validity checks, so §3's semantic "
                  "comparison is not run and the PI gets a human-checkable set instead."})
    print(f"wrote {len(rows)} strips to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
