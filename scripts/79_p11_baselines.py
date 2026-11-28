"""P11: six-way comparison on the official ChordEdit base at a hard cap of 5 UNet forwards.

Methods (all reimplemented from the papers -- none of the three has runnable official code):

``official``      ChordEdit's own chord transport + ``t_c=0.3`` prox
``erchord_mu{1,2}`` ER-Chord: per-pixel cross-scale cosine -> ``w = 1/(1+mu(1-c)^2)`` -> equal energy
``budedit``       BudEdit: attention-selected support, energy budget, gated refinement
``precise_gamma`` PreciseEdit: SCT + directional gate + PreciseMask + the paper's composition
``precise_a1``    the same with ``gamma = 1`` (fairness row: SCT without the gate)
``scadA_a1``      our SCAD-A, now at exactly 5 forwards (the guided t=0.75 call *is* the field)
``scad_adapt``    SCAD with a per-image alpha from the in-mask response ratio
``scad_cleanbg``  SCAD with a wide layer mask and a tighter output mask

Every variant runs its own 5-forward pipeline (2 at t=0.9, 1 src at t=0.75, 1 tar at t=0.75, 1 prox),
so the budget is identical across methods and nothing is shared or precomputed.
"""

from __future__ import annotations

import argparse
import os
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

from geoedit import baselines as bl  # noqa: E402
from geoedit import inlayer  # noqa: E402
from geoedit.chord import ChordEditor  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now  # noqa: E402
from geoedit.masks import gaussian_blur  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402

FIELDS = ["image_id", "category", "variant", "nfe", "psnr_unedit", "lpips_unedit", "ssim_unedit",
          "struct_dist", "struct_dist_unedit", "clip_target_edit", "mask_area", "mask_in",
          "mask_out", "extra"]
SCORE_KEYS = FIELDS[4:10]
T_H, T_L, T_C = 0.9, 0.75, 0.3
W_H = 0.15 / 1.05


def x0(unet, z, t_idx, a, b, ctx):
    ts = torch.full((z.shape[0],), int(t_idx), device=z.device, dtype=torch.long)
    eps = unet(sample=z, timestep=ts, encoder_hidden_states=ctx, return_dict=False)[0]
    return (z - float(b) * eps) / float(a)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--p11-config", default="configs/p11.yaml")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()
    base = load_yaml(ROOT / args.config)
    p11 = load_yaml(ROOT / args.p11_config)
    specs: Dict[str, dict] = {v["name"]: v for v in p11["variants"]}
    names = list(specs) if not args.variants else [n for n in args.variants.split(",") if n in specs]

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = list(range(len(dataset)))
    eval_idx = [pool[i] for i in np.linspace(0, len(pool) - 1, int(p11["n_images"]))
                .round().astype(int).tolist()]
    indices = [i for k, i in enumerate(eval_idx) if k % args.num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]

    tag = "p11_700" if "700" in Path(args.p11_config).stem else "p11"
    out_dir = ROOT / base["output_root"] / tag
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
    tok = ed.pipe.tokenizer
    lo, hi = float(p11["mask"]["q_lo"]), float(p11["mask"]["q_hi"])
    sig = float(p11["mask"]["sigma"])
    a_max, r0 = float(p11["scad_adapt"]["alpha_max"]), float(p11["scad_adapt"]["r0"])
    print(f"[p11 shard {args.shard}/{args.num_shards}] {len(indices)} images, "
          f"{len(names)} variants on {args.device}", flush=True)

    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        todo = [n for n in names if (sample.image_id, n) not in done]
        if not todo:
            continue
        x_s = ed.encode(sample.image()).float()
        ctx_s, ctx_t = ed.encode_prompt(sample.src_prompt), ed.encode_prompt(sample.tar_prompt)
        noise = ed.pipe._prepare_noise_list(x_s, 42, 1)[0].float()
        idx = lambda t: int(round(ed.pipe._max_unet_timestep * t))
        a_h, b_h = ed.pipe._get_alpha_sigma(x_s, ed.pipe._time_to_index(1, T_H, device=x_s.device))
        a_l, b_l = ed.pipe._get_alpha_sigma(x_s, ed.pipe._time_to_index(1, T_L, device=x_s.device))
        a_c, b_c = ed.pipe._get_alpha_sigma(x_s, ed.pipe._time_to_index(1, T_C, device=x_s.device))
        e_src = inlayer.edit_token_indices(tok, sample.src_prompt, sample.tar_prompt)
        e_tar = e_src

        rows = []
        for name in todo:
            spec = specs[name]
            method = spec["method"]
            state.active = False
            state.capture2 = state.capture1 = False
            state.acc2, state.acc1 = [], None
            with torch.no_grad():
                # ---- t = 0.9 (no guidance) ----
                z_h = float(a_h) * x_s + float(b_h) * noise
                want_r = method == "scad_adapt"
                if want_r:
                    state.clear_capture()
                    state.capture_o = "src"
                mu_h_s = x0(ed.pipe.unet, z_h, idx(T_H), a_h, b_h, ctx_s)
                state.capture_o = "tar" if want_r else None
                mu_h_t = x0(ed.pipe.unet, z_h, idx(T_H), a_h, b_h, ctx_t)
                state.capture_o = None
                r_val = float((state.r_num / max(state.r_den, 1e-8)) ** 0.5) if want_r else 0.0
                R_h, A_h = mu_h_t - mu_h_s, mu_h_t - x_s
                E_h = gaussian_blur(R_h[0].norm(dim=0), sig)
                M_h = inlayer.plateau_mask(E_h, lo, hi)

                # ---- t = 0.75: src (capture when a baseline needs the attention map) ----
                need_phi = method in ("budedit", "precise")
                z_l = float(a_l) * x_s + float(b_l) * noise
                if need_phi:
                    state.capture2 = True
                    state.edit_idx = e_src
                mu_l_s = x0(ed.pipe.unet, z_l, idx(T_L), a_l, b_l, ctx_s)
                phi_src = bl.attention_map(state.acc2) if need_phi else None
                state.capture2 = False
                state.acc2 = []

                # ---- t = 0.75: tar (guided for SCAD, captured for the baselines) ----
                if method in ("scad", "scad_adapt", "scad_cleanbg"):
                    alpha = float(spec.get("alpha", 1.0))
                    if method == "scad_adapt":
                        alpha = bl.adaptive_alpha(r_val, a_max, r0)
                    inlayer.set_call(state, src_ctx=ctx_s, mask=M_h, scad="A", alpha=alpha)
                    mu_l_t = x0(ed.pipe.unet, z_l, idx(T_L), a_l, b_l, ctx_t)
                    state.active = False
                else:
                    if need_phi:
                        state.capture2 = True
                        state.edit_idx = e_tar
                    if method == "precise":
                        state.capture1 = True
                    mu_l_t = x0(ed.pipe.unet, z_l, idx(T_L), a_l, b_l, ctx_t)
                    phi_tar = bl.attention_map(state.acc2) if need_phi else None
                    attn1 = state.acc1 if method == "precise" else None
                    state.capture2 = state.capture1 = False
                    state.acc2, state.acc1 = [], None
                R_l, A_l = mu_l_t - mu_l_s, mu_l_t - x_s
                E_l = gaussian_blur(R_l[0].norm(dim=0), sig)
                M_l = inlayer.plateau_mask(E_l, lo, hi)
                u_chord = W_H * R_h + (1.0 - W_H) * R_l

                # ---- build the prox input and the final output per method ----
                extra = ""
                if method == "official":
                    y = x_s + u_chord
                    prox = x0(ed.pipe.unet, float(a_c) * y + float(b_c) * noise, idx(T_C), a_c, b_c, ctx_t)
                    out = prox
                elif method.startswith("erchord"):
                    u_er, c = bl.er_chord_field(u_chord, R_h, R_l, mu=float(spec["mu"]))
                    y = x_s + u_er
                    prox = x0(ed.pipe.unet, float(a_c) * y + float(b_c) * noise, idx(T_C), a_c, b_c, ctx_t)
                    out = prox
                    extra = f"cos_mean={float(c.mean()):.3f}"
                elif method == "budedit":
                    phi = torch.maximum(phi_src.to(x_s.device), phi_tar.to(x_s.device))
                    delta, phi_pin = bl.budedit_field(u_chord, A_l, phi, beta=float(spec["beta"]))
                    y = x_s + delta
                    prox = x0(ed.pipe.unet, float(a_c) * y + float(b_c) * noise, idx(T_C), a_c, b_c, ctx_t)
                    out = (1.0 - phi_pin) * x_s + phi_pin * prox
                elif method == "precise":
                    k = float(spec["k"])
                    u_h_sct = R_h + k * A_h
                    u_l_sct = R_l + k * A_l
                    gam = bl.cosine_gate(u_h_sct, u_l_sct) if spec.get("gate", True) else 1.0
                    u_sct = (u_chord + k * A_l) / (1.0 + k)
                    phi = torch.maximum(phi_src.to(x_s.device), phi_tar.to(x_s.device))
                    M_p = bl.precise_mask(phi, attn1, beta_lap=0.5)
                    x_t = x_s + M_p.unsqueeze(0) * (gam * u_sct)
                    prox = x0(ed.pipe.unet, float(a_c) * x_t + float(b_c) * noise, idx(T_C), a_c, b_c, ctx_t)
                    out = bl.precise_compose(x_s, u_sct, M_p, gam, prox)
                    extra = f"gamma={gam:.3f}"
                elif method in ("scad", "scad_adapt", "scad_cleanbg"):
                    y = x_s + u_chord
                    inlayer.set_call(state, src_ctx=ctx_s, mask=M_l, scad="A", alpha=alpha)
                    prox = x0(ed.pipe.unet, float(a_c) * y + float(b_c) * noise, idx(T_C), a_c, b_c, ctx_t)
                    state.active = False
                    M_out = M_l if method != "scad_cleanbg" else inlayer.plateau_mask(E_l, 0.50, 0.70)
                    out = (1.0 - M_out) * x_s + M_out * prox
                    extra = f"alpha={alpha:.3f}" + (f" r={r_val:.3f}" if method == "scad_adapt" else "")
                else:
                    raise ValueError(method)
                state.active = False
            rows.append({"image_id": sample.image_id, "category": sample.category,
                         "variant": name, "nfe": 5,
                         "mask_area": float((M_l > 0.5).float().mean()),
                         "mask_in": float(M_l[torch.from_numpy(sample.mask64 > 0.5)].mean()),
                         "mask_out": float(M_l[torch.from_numpy(sample.mask64 <= 0.5)].mean()),
                         "extra": extra, "_latent": out})
        decoded = [ed.decode(r["_latent"])[0] for r in rows]
        source = np.asarray(sample.image())
        scored = metrics.evaluate_many(source, decoded, sample.mask512, sample.src_prompt,
                                       sample.tar_prompt, chunk=4)
        for r, s, img in zip(rows, scored, decoded):
            r.update({k: s[k] for k in SCORE_KEYS})
            r.pop("_latent")
            if args.dump:
                from PIL import Image
                d = ROOT / "cache" / "images" / "p11" / r["variant"]
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(d / f"{sample.image_id}.png")
        append_rows(csv_path, rows, FIELDS)
        processed += 1
        if os.environ.get("P11_MEM"):
            print(f"    mem {torch.cuda.memory_allocated()/2**30:.2f}GB "
                  f"res {torch.cuda.memory_reserved()/2**30:.2f}GB", flush=True)
        if processed % 5 == 0:
            torch.cuda.empty_cache()
        if processed % 10 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / processed:.1f}s/img eta "
                  f"{(len(indices) - position - 1) * el / processed / 60:.1f} min", flush=True)
    print(f"[p11 shard {args.shard}] done={processed} in "
          f"{(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
