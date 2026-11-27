"""P6 Stage 1: the visual k-sweep on the 50 pairs the PI already looked at.

Same 50 images as ``results/p5/pairs/p5_pairs_manifest.csv`` (fixed seed 20261126), so the
comparison is against the exact strips that motivated the SDS term in the first place:

    source | k=0 (plain masked DDS) | k=1 | k=2 | k=2, gamma=2 (core-concentrated)

at a fixed ``lam = 1`` (the P5 strips were lam=1 as well, so the "orange cat stays orange" case is
directly comparable).  Each panel is annotated with the background PSNR, and the strip carries the
instruction.  No UNet forward: everything comes from the P1 residuals + the ``mu_src`` cache.

Outputs ``results/p6/vis_k/<image_id>.png`` + ``p6_vis_k.csv`` (+ two contact sheets).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.data import PIEDataset  # noqa: E402
from geoedit.edits import attention_term, l5_mask, ours4_delta  # noqa: E402
from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

# (label, k, gamma)
VARIANTS = [("k=0", 0.0, 1.0), ("k=1", 1.0, 1.0), ("k=2", 2.0, 1.0), ("k=2 g=2", 2.0, 2.0)]


def _font(size: int):
    for c in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p6-config", default="configs/p6.yaml")
    parser.add_argument("--manifest", default="results/p5/pairs/p5_pairs_manifest.csv")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--contact", type=int, default=6, help="images per contact sheet")
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    p6 = load_yaml(ROOT / args.p6_config)

    manifest = pd.read_csv(ROOT / args.manifest, dtype={"image_id": str, "category": str})
    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    by_id = {dataset[i].image_id: i for i in range(len(dataset))}
    a_root = ROOT / base["output_root"] / "p3" / "a"
    mu_root = ROOT / base["output_root"] / "p6" / "a" / "musrc"
    out_dir = ROOT / base["output_root"] / "p6" / "vis_k"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision="fp32",
                    text_precision="fp32", vae_only=True)
    font, small = _font(20), _font(15)
    sigma = float(p6.get("sigma_4nfe", 2.0))

    rows: List[dict] = []
    for n, rec in manifest.iterrows():
        image_id = rec["image_id"]
        sample = dataset[by_id[image_id]]
        x_s = torch.load(a_root / "xs" / f"{image_id}.pt", weights_only=True).float().to(model.device)
        x_s = x_s[0] if x_s.dim() == 4 else x_s
        bundle = torch.load(a_root / "R" / f"{image_id}.pt", weights_only=True)
        R9 = bundle["t0.9_s0"].float().to(model.device)
        R5 = bundle["t0.5_s0"].float().to(model.device)
        mu9 = torch.load(mu_root / f"{image_id}.pt", weights_only=True)["t0.9_s0"].float().to(model.device)
        A9 = attention_term(R9, mu9, x_s)
        M = l5_mask(R5, sigma)

        latents = {"source": x_s.unsqueeze(0)}
        meta: Dict[str, dict] = {}
        for label, k, gamma in VARIANTS:
            delta = ours4_delta(R9, A9, M, args.lam, k, gamma)
            latents[label] = (x_s + delta).unsqueeze(0)
            meta[label] = {"k": k, "gamma": gamma}
        decoded = {}
        for label, lat in latents.items():
            decoded[label] = model.decode(lat)[0]

        from geoedit.metrics import PieMetrics  # heavy, import once
        if not hasattr(main, "_metrics"):
            main._metrics = PieMetrics(device=args.device)  # type: ignore[attr-defined]
        metrics = main._metrics  # type: ignore[attr-defined]
        source = np.asarray(sample.image())
        labels = list(latents.keys())
        scored = metrics.evaluate_many(source, [decoded[l] for l in labels], sample.mask512,
                                       sample.src_prompt, sample.tar_prompt, chunk=5)

        n_panel = len(labels)
        strip = Image.new("RGB", (512 * n_panel, 512 + 78), (255, 255, 255))
        draw = ImageDraw.Draw(strip)
        for j, label in enumerate(labels):
            strip.paste(Image.fromarray(decoded[label]), (512 * j, 0))
            draw.text((512 * j + 8, 512 + 2), label, fill=(0, 0, 0), font=font)
            if label != "source":
                draw.text((512 * j + 8, 512 + 26),
                          f"PSNR {scored[j]['psnr_unedit']:.1f} dB", fill=(60, 60, 60), font=small)
                rows.append({"image_id": image_id, "category": sample.category, "variant": label,
                             "k": meta[label]["k"], "gamma": meta[label]["gamma"], "lam": args.lam,
                             "psnr_unedit": scored[j]["psnr_unedit"],
                             "lpips_unedit": scored[j]["lpips_unedit"],
                             "struct_dist_unedit": scored[j]["struct_dist_unedit"],
                             "clip_target_edit": scored[j]["clip_target_edit"]})
        draw.text((8, 512 + 50), f"{image_id} (cat {sample.category})  instruction: "
                                 f"{sample.instruction}", fill=(0, 0, 0), font=small)
        strip.save(out_dir / f"{image_id}.png")
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(manifest)}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "p6_vis_k.csv", index=False)
    write_json(out_dir / "p6_vis_k_meta.json", {
        "created": now(), "n_images": int(manifest["image_id"].nunique()), "lam": args.lam,
        "sigma": sigma, "variants": [v[0] for v in VARIANTS], "device": args.device,
        "note": "same 50 images as results/p5/pairs (seed 20261126); k=0 is the plain masked DDS "
                "transport at the same lam, so the only change is the in-mask attraction term."})

    # contact sheet: medians of psnr per variant + the mean of the 5-panel strips
    if args.contact:
        picks = manifest["image_id"].tolist()[: args.contact]
        w, h = 512 * 5, 512 + 78
        sheet = Image.new("RGB", (w, h * len(picks)), (255, 255, 255))
        for i, image_id in enumerate(picks):
            sheet.paste(Image.open(out_dir / f"{image_id}.png"), (0, h * i))
        sheet.save(out_dir / "contact_sheet.png")

    summary = (table.groupby(["variant", "k", "gamma"])[
        ["psnr_unedit", "lpips_unedit", "struct_dist_unedit", "clip_target_edit"]].mean().round(3))
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
