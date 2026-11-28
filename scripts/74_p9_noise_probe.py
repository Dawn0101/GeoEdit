"""P9 step 1: single-factor probe for the CLIP-Edited gap between our ChordEdit replica and the
official B1_chord run, on exactly the same images and metrics.

Factors checked analytically (see REPORT_p9.md): image preprocessing (all 200 eval images are 512x512,
so ChordEdit's centre-crop transform is a no-op), VAE latent_dist.mode() * scaling_factor (both),
timestep round(t * (num_train_timesteps - 1)) (both), prompt text (PIE-Bench's raw prompts carry no []
for this benchmark -- identical token streams), and the evaluation subset (chord on the same 137
images: 22.41 vs 22.59 on all 200, so the subset is worth only 0.18).

The one genuinely different factor is the NOISE DRAW: ChordEdit calls torch.manual_seed(42) +
torch.randn_like(..., device='cuda') -- one fixed tensor for every image -- while we draw per image on
the CPU from sha256(image_id|seed).  This script swaps that single factor and rebuilds the whole chord
field (R(0.9) and R(0.75)) for the alternative draw.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import edit_ops  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.edits import prox_latent  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

FIELDS = ["image_id", "category", "variant", "lam", "psnr_unedit", "lpips_unedit", "ssim_unedit",
          "struct_dist", "struct_dist_unedit", "clip_target_edit", "u_norm"]
W_CHORD, T_H, T_L, PROX_T = 0.143, 0.9, 0.75, 0.3


def chord_field_with_noise(model, x_s, emb_src, emb_tar, eps):
    """Chord field for one noise draw (4 UNet forwards): 0.143*R(0.9) + 0.857*R(0.75)."""
    res = {}
    sq = x_s.dim() == 3
    xs = x_s.unsqueeze(0) if sq else x_s
    es = eps.unsqueeze(0) if eps.dim() == 3 else eps
    for t in (T_H, T_L):
        a, b = model.ab(t)
        z = float(a) * xs + float(b) * es
        r = model.x0_pred(z, t, emb_tar) - model.x0_pred(z, t, emb_src)
        if not sq:
            res[t] = r
        else:
            res[t] = r[0]
    return W_CHORD * res[T_H] + (1.0 - W_CHORD) * res[T_L]
    return W_CHORD * res[T_H] + (1.0 - W_CHORD) * res[T_L]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = list(range(len(dataset)))
    eval_idx = [pool[i] for i in np.linspace(0, len(pool) - 1, 200).round().astype(int).tolist()]
    picked = eval_idx[: args.n]

    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision="fp32",
                    text_precision="fp32")
    metrics = PieMetrics(device=args.device)
    a_root = ROOT / base["output_root"] / "p3" / "a"
    t75_root = ROOT / base["output_root"] / "p8" / "a" / "t75"
    out_dir = ROOT / base["output_root"] / "p9" / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "p9_probe.csv"
    print(f"[p9 probe] {len(picked)} images, lam={args.lam}, device={args.device}", flush=True)

    for position, index in enumerate(picked):
        sample = dataset[index]
        x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt", weights_only=True).float().to(model.device)
        x_s = x_s[0] if x_s.dim() == 4 else x_s
        emb_src = model.embed([sample.src_prompt])
        emb_tar = model.embed([sample.tar_prompt])

        R9 = torch.load(a_root / "R" / f"{sample.image_id}.pt",
                        weights_only=True)["t0.9_s0"].float().to(model.device)
        d75 = torch.load(t75_root / f"{sample.image_id}.pt", weights_only=True)
        R75 = (d75["R75"][0] if d75["R75"].dim() == 4 else d75["R75"]).float().to(model.device)
        eps_hash = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
        u_hash = W_CHORD * R9 + (1.0 - W_CHORD) * R75

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        eps_chord = torch.randn_like(x_s, device=x_s.device, dtype=torch.float32)
        with torch.no_grad():
            u_chord = chord_field_with_noise(model, x_s, emb_src, emb_tar, eps_chord)
            latents: List[torch.Tensor] = []
            names: List[str] = []
            u_norms: List[float] = []
            for name, u, eps in (("hash_noise", u_hash, eps_hash),
                                 ("chord_noise42", u_chord, eps_chord)):
                latents.append(prox_latent(model, x_s + args.lam * u, PROX_T, emb_tar, eps))
                names.append(name)
                u_norms.append(float(u.norm()))
        decoded = [model.decode(lat.unsqueeze(0))[0] for lat in latents]
        source = np.asarray(sample.image())
        scored = metrics.evaluate_many(source, decoded, sample.mask512, sample.src_prompt,
                                       sample.tar_prompt, chunk=4)
        rows = [{"image_id": sample.image_id, "category": sample.category, "variant": n,
                 "lam": args.lam, **{k: s[k] for k in FIELDS[4:-1]}, "u_norm": un}
                for n, s, un in zip(names, scored, u_norms)]
        append_rows(csv_path, rows, FIELDS)
        if (position + 1) % 20 == 0:
            print(f"  {position + 1}/{len(picked)}", flush=True)

    frame = pd.read_csv(csv_path, dtype={"image_id": str}).drop_duplicates(subset=["image_id", "variant"])
    print(f"\n[p9 probe] done ({now()}); n={frame.image_id.nunique()}")
    print(frame.groupby("variant")[["psnr_unedit", "lpips_unedit", "struct_dist",
                                    "clip_target_edit", "u_norm"]].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
