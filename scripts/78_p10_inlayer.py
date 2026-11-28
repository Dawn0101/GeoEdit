"""P10: in-layer spatial guidance on the **official ChordEdit pipeline**, 5 UNet forwards, 200 images.

Why in-layer: the post-hoc soft mask was multiplied twice (transport *and* final composite) and its
mean inside the object was 0.61, so only ~0.37 of the edit survived; and multiplying afterwards can
only *cut* the background, never strengthen the semantics inside the object.  The mask now enters the
cross-attention of the target branch, before the velocity prediction.

Pipeline per image (exactly 5 forwards, same as ChordEdit):

``t=0.9``  src call + tar call (no hooks)                  -> R(0.9)  -> M_0.9  (plateau)
``t=0.75`` src call (no hooks) + tar call (**hooks**, M_0.9) -> R(0.75) -> M_0.75
``u``      ``0.143*R(0.9) + 0.857*R(0.75)``                 (ChordEdit's chord weights)
``prox``   tar call at ``t_c=0.3`` (**hooks**, M_0.75)
output     ``prox``  or  ``(1-M)*x_s + M*prox``  depending on the variant

Variants (see ``configs/p10.yaml``): the official baseline, the two single-gate plateau variants,
SCAD-A/B, MGA at three strengths (with and without the post mask), and SCAD+MGA together.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import inlayer  # noqa: E402
from geoedit.chord import ChordEditor  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now  # noqa: E402
from geoedit.masks import gaussian_blur  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402

FIELDS = ["image_id", "category", "variant", "nfe", "psnr_unedit", "lpips_unedit", "ssim_unedit",
          "struct_dist", "struct_dist_unedit", "clip_target_edit", "mask_area", "mask_in",
          "mask_out", "edit_tokens"]
SCORE_KEYS = FIELDS[4:10]
T_H, T_L, T_C = 0.9, 0.75, 0.3
W_H = 0.15 / 1.05  # 0.1429


def x0(unet, z, t_idx, a, b, ctx):
    ts = torch.full((z.shape[0],), int(t_idx), device=z.device, dtype=torch.long)
    eps = unet(sample=z, timestep=ts, encoder_hidden_states=ctx, return_dict=False)[0]
    return (z - float(b) * eps) / float(a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--p10-config", default="configs/p10.yaml")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", default=None, help="comma separated subset")
    ap.add_argument("--dump", action="store_true", help="also save the decoded images")
    args = ap.parse_args()
    base = load_yaml(ROOT / args.config)
    p10 = load_yaml(ROOT / args.p10_config)
    specs: Dict[str, dict] = {v["name"]: v for v in p10["variants"]}
    names = list(specs) if not args.variants else [n for n in args.variants.split(",") if n in specs]

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = list(range(len(dataset)))
    eval_idx = [pool[i] for i in np.linspace(0, len(pool) - 1, int(p10["n_images"]))
                .round().astype(int).tolist()]
    indices = [i for k, i in enumerate(eval_idx) if k % args.num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]

    out_dir = ROOT / base["output_root"] / "p10"
    (out_dir / "frontier").mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "frontier" / (f"per_image_shard{args.shard}.csv" if args.num_shards > 1
                                       else "per_image.csv")
    done = set()
    if csv_path.exists() and not args.dump:
        prev = pd.read_csv(csv_path, dtype={"image_id": str}, on_bad_lines="skip")
        done = set(zip(prev["image_id"], prev["variant"]))

    ed = ChordEditor(device=args.device)
    state = inlayer.install(ed.pipe.unet)
    metrics = PieMetrics(device=args.device)
    tokenizer = ed.pipe.tokenizer
    lo, hi = float(p10["mask"]["q_lo"]), float(p10["mask"]["q_hi"])
    sigma = float(p10["mask"]["sigma"])
    print(f"[p10 shard {args.shard}/{args.num_shards}] {len(indices)} images, "
          f"{len(names)} variants on {args.device}", flush=True)

    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        todo = [n for n in names if (sample.image_id, n) not in done]
        if not todo:
            continue
        image = sample.image()
        x_s = ed.encode(image).float()
        ctx_s = ed.encode_prompt(sample.src_prompt)
        ctx_t = ed.encode_prompt(sample.tar_prompt)
        noise = ed.pipe._prepare_noise_list(x_s, 42, 1)[0].float()
        a_h, b_h = ed.pipe._get_alpha_sigma(x_s, ed.pipe._time_to_index(1, T_H, device=x_s.device))
        a_l, b_l = ed.pipe._get_alpha_sigma(x_s, ed.pipe._time_to_index(1, T_L, device=x_s.device))
        t_h_idx = int(round(ed.pipe._max_unet_timestep * T_H))
        t_l_idx = int(round(ed.pipe._max_unet_timestep * T_L))
        a_c, b_c = ed.pipe._get_alpha_sigma(x_s, ed.pipe._time_to_index(1, T_C, device=x_s.device))
        t_c_idx = int(round(ed.pipe._max_unet_timestep * T_C))

        # ---- t = 0.9 (no hooks): the first step only produces the coarse mask ----
        state.active = False
        z_s = x_s + (float(b_h) / float(a_h)) * noise * 0.0  # placeholder to keep shapes explicit
        z_h = (float(a_h) * x_s + float(b_h) * noise)
        with torch.no_grad():
            mu_h_src = x0(ed.pipe.unet, z_h, t_h_idx, a_h, b_h, ctx_s)
            mu_h_tar = x0(ed.pipe.unet, z_h, t_h_idx, a_h, b_h, ctx_t)
        R_h = (mu_h_tar - mu_h_src)
        E_h = gaussian_blur(R_h[0].norm(dim=0), sigma)
        M_h = inlayer.plateau_mask(E_h, lo, hi)

        # ---- t = 0.75 (tar call guided by M_h) ----
        z_l = (float(a_l) * x_s + float(b_l) * noise)
        with torch.no_grad():
            mu_l_src = x0(ed.pipe.unet, z_l, t_l_idx, a_l, b_l, ctx_s)
            inlayer.set_call(state, src_ctx=ctx_s, mask=M_h, scad=None, mga=0.0)  # off for the field
            mu_l_tar_plain = x0(ed.pipe.unet, z_l, t_l_idx, a_l, b_l, ctx_t)
        R_l = (mu_l_tar_plain - mu_l_src)
        E_l = gaussian_blur(R_l[0].norm(dim=0), sigma)
        M_l = inlayer.plateau_mask(E_l, lo, hi)

        # ---- per-variant transport + prox ----
        rows = []
        for name in todo:
            spec = specs[name]
            hook = spec.get("hook")
            alpha = float(spec.get("alpha", 0.0))
            mga = float(spec.get("mga", 0.0))
            edit_idx = inlayer.edit_token_indices(tokenizer, sample.src_prompt, sample.tar_prompt) \
                if hook in ("MGA", "AM") else []
            # transport: re-run the t=0.75 tar branch with the hooks, guided by M_h
            state.active = False
            with torch.no_grad():
                if hook in ("SCAD_A", "SCAD_B", "MGA", "AM"):
                    inlayer.set_call(state, src_ctx=ctx_s, mask=M_h,
                                     scad="A" if hook in ("SCAD_A", "AM") else
                                          ("B" if hook == "SCAD_B" else None),
                                     alpha=alpha, mga=mga if hook in ("MGA", "AM") else 0.0,
                                     edit_idx=edit_idx)
                    mu_l_tar = x0(ed.pipe.unet, z_l, t_l_idx, a_l, b_l, ctx_t)
                    state.active = False
                else:
                    mu_l_tar = mu_l_tar_plain
                u = W_H * R_h + (1.0 - W_H) * (mu_l_tar - mu_l_src)
                delta = M_l.unsqueeze(0) * u[0] if spec.get("trans_gate") else u[0]
                y = x_s + delta.unsqueeze(0)
                # prox, guided by M_l
                if hook in ("SCAD_A", "SCAD_B", "MGA", "AM"):
                    inlayer.set_call(state, src_ctx=ctx_s, mask=M_l,
                                     scad="A" if hook in ("SCAD_A", "AM") else
                                          ("B" if hook == "SCAD_B" else None),
                                     alpha=alpha, mga=mga if hook in ("MGA", "AM") else 0.0,
                                     edit_idx=edit_idx)
                z_c = float(a_c) * y + float(b_c) * noise
                prox = x0(ed.pipe.unet, z_c, t_c_idx, a_c, b_c, ctx_t)
                state.active = False
            lat = (1.0 - M_l) * x_s + M_l * prox if spec.get("out_gate") else prox
            rows.append({"image_id": sample.image_id, "category": sample.category,
                         "variant": name, "nfe": 5,
                         "mask_area": float((M_l > 0.5).float().mean()),
                         "mask_in": float(M_l[torch.from_numpy(sample.mask64 > 0.5)].mean()),
                         "mask_out": float(M_l[torch.from_numpy(sample.mask64 <= 0.5)].mean()),
                         "edit_tokens": " ".join(map(str, edit_idx)),
                         "_latent": lat})
        decoded = [ed.decode(r["_latent"])[0] for r in rows]
        if args.dump:
            from PIL import Image
            for r, img in zip(rows, decoded):
                d = ROOT / "cache" / "images" / "p10" / r["variant"]
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(d / f"{sample.image_id}.png")
        source = np.asarray(sample.image())
        scored = metrics.evaluate_many(source, decoded, sample.mask512, sample.src_prompt,
                                       sample.tar_prompt, chunk=4)
        for r, s in zip(rows, scored):
            r.update({k: s[k] for k in SCORE_KEYS})
            r.pop("_latent", None)
        append_rows(csv_path, rows, FIELDS)
        processed += 1
        if processed % 5 == 0:
            torch.cuda.empty_cache()
        if processed % 10 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / processed:.1f}s/img eta "
                  f"{(len(indices) - position - 1) * el / processed / 60:.1f} min", flush=True)
    print(f"[p10 shard {args.shard}] done={processed} in "
          f"{(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
