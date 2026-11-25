"""Stage A sanity checks (S1-S4) before any experiment runs.

S1  R == 0 when source and target prompt are identical          (T-M3)
S2  same (image_id, seed) twice gives bit-identical results      (T-M4)
S3  fp16 vs fp32 residual: median relative difference < 1%       (gate for fp16)
S4  audit gate on N images: lam=1 + prox at t=0.9 must land in
    PSNR [20, 27] and CLIP-Edited >= 20                          (§4.4)

Writes ``results/sanity.json`` and ``results/sanity.log``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from geoedit import edit_ops
from geoedit.data import PIEDataset
from geoedit.io_utils import env_snapshot, load_yaml, now, write_json
from geoedit.metrics import PieMetrics
from geoedit.model import SDTurbo

ROOT = Path(__file__).resolve().parents[1]


def nu_table(model: SDTurbo, ts=(0.3, 0.5, 0.7, 0.9)) -> list:
    rows = []
    for t in ts:
        a, b = model.ab(t)
        rows.append({"t": t, "timestep": model.timestep_index(t), "a": round(a, 5),
                     "b": round(b, 5), "nu": round((b / a) ** 2, 3)})
    return rows


def check_s1(model: SDTurbo, dataset: PIEDataset, n: int = 5, t: float = 0.9) -> dict:
    worst = 0.0
    worst_id = None
    for sample in list(dataset)[:n]:
        image = sample.image()
        x_s = model.encode([image])
        eps = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
        emb = model.embed([sample.src_prompt])
        res = edit_ops.residual(model, x_s, t, emb, emb, eps)
        value = float(res.abs().max().item())
        if value > worst:
            worst, worst_id = value, sample.image_id
    return {"pass": worst < 1e-3, "max_abs_R": worst, "worst_image": worst_id,
            "note": "prompt identical -> R must vanish; fp16 tolerance 1e-3"}


def check_s2(model: SDTurbo, dataset: PIEDataset, t: float = 0.9) -> dict:
    sample = list(dataset)[0]
    x_s = model.encode([sample.image()])
    eps = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
    emb_src = model.embed([sample.src_prompt])
    emb_tar = model.embed([sample.tar_prompt])
    first = edit_ops.residual(model, x_s, t, emb_src, emb_tar, eps)
    second = edit_ops.residual(model, x_s, t, emb_src, emb_tar, eps)
    same = bool(torch.equal(first, second))
    return {"pass": same, "bitwise_equal": same, "image": sample.image_id}


def check_s3(dataset: PIEDataset, n: int = 20, t: float = 0.9, device: str = "cuda:0",
             base_precision: str = "fp16") -> dict:
    """fp16 vs fp32 UNet with the *same* fp32 noise; median relative norm difference."""
    import torch

    m16 = SDTurbo(device=device, unet_precision="fp16", vae_precision="fp16")
    m32 = SDTurbo(device=device, unet_precision="fp32", vae_precision="fp32")
    rel = []
    for sample in list(dataset)[:n]:
        x_s = m16.encode([sample.image()])
        eps32 = edit_ops.make_eps(sample.image_id, 0, m16.device, torch.float32)
        emb_src = m16.embed([sample.src_prompt])
        emb_tar = m16.embed([sample.tar_prompt])
        r16 = edit_ops.residual(m16, x_s.to(torch.float16), t, emb_src, emb_tar, eps32.to(torch.float16))
        r32 = edit_ops.residual(m32, x_s.float(), t, emb_src.float(), emb_tar.float(), eps32)
        num = float((r16.float() - r32).norm().item())
        den = float(r32.norm().item())
        rel.append(num / max(den, 1e-8))
    median = float(np.median(rel))
    del m32
    torch.cuda.empty_cache()
    # The gate is "use fp16 only if it matches fp32 to 1%".  Once the config runs in fp32
    # the comparison is informational, so the check passes by construction.
    using_fp32 = base_precision == "fp32"
    return {"pass": bool(median < 0.01 or using_fp32), "median_rel": median,
            "max_rel": float(np.max(rel)), "n": len(rel), "using_fp32": using_fp32,
            "note": "median_rel < 1% required to keep fp16; otherwise run everything in fp32"}


def check_s4(model: SDTurbo, dataset: PIEDataset, metrics: PieMetrics, args, n: int = 50,
             t: float = 0.9, lam: float = 1.0) -> dict:
    from geoedit import edit_ops as ops

    # Even stride over the 700 samples so the audit covers all 10 edit categories.
    indices = np.linspace(0, len(dataset) - 1, num=n).round().astype(int)
    rows = []
    for sample in [dataset[int(i)] for i in indices]:
        image = sample.image()
        x_s = model.encode([image])
        eps = ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
        eps_c = ops.make_eps(sample.image_id, args.seed_base + args.prox_seed_offset,
                             model.device, model.unet_dtype)
        emb_src = model.embed([sample.src_prompt])
        emb_tar = model.embed([sample.tar_prompt])
        res = ops.residual(model, x_s, t, emb_src, emb_tar, eps)
        y = ops.lam_edit(x_s, res, lam)
        y_prox = ops.prox(model, y, args.t_c, emb_tar, eps_c)
        out_edit = model.decode(y_prox)[0]
        out_recon = model.decode(x_s)[0]
        src = np.asarray(image)
        edited = metrics.evaluate(src, out_edit, sample.mask512, sample.src_prompt, sample.tar_prompt)
        recon = metrics.evaluate(src, out_recon, sample.mask512, sample.src_prompt, sample.tar_prompt)
        rows.append({"image_id": sample.image_id, "category": sample.category,
                     "psnr": edited["psnr_unedit"], "clip_edited": edited["clip_edited"],
                     "lpips": edited["lpips_unedit"], "struct_dist": edited["struct_dist"],
                     "recon_psnr": recon["psnr_unedit"]})
        print(f"  {sample.image_id} cat={sample.category} "
              f"PSNR={edited['psnr_unedit']:.2f} CLIP-E={edited['clip_edited']:.2f} "
              f"recon PSNR={recon['psnr_unedit']:.2f}", flush=True)

    psnr = float(np.nanmean([r["psnr"] for r in rows]))
    clip = float(np.nanmean([r["clip_edited"] for r in rows]))
    recon = float(np.nanmean([r["recon_psnr"] for r in rows]))
    lo, hi = args.audit_psnr_range
    n_valid = int(np.sum(np.isfinite([r["psnr"] for r in rows])))
    return {
        "pass": bool(lo <= psnr <= hi and clip >= args.audit_clip_min and recon > psnr),
        "n_valid_psnr": n_valid,
        "mean_psnr_unedit": psnr,
        "mean_clip_edited": clip,
        "mean_recon_psnr": recon,
        "psnr_range": [lo, hi],
        "clip_min": args.audit_clip_min,
        "n": len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p2-config", default="configs/p2_howfar.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", default="results/sanity.json")
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p2 = load_yaml(ROOT / args.p2_config)
    args.seed_base = base["seed_base"]
    args.prox_seed_offset = p2["prox_seed_offset"]
    args.t_c = p2["t_c"]
    args.audit_psnr_range = p2["audit_psnr_range"]
    args.audit_clip_min = p2["audit_clip_min"]

    log_path = ROOT / "results" / "sanity.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a") as handle:
            handle.write(f"[{now()}] {message}\n")

    log("=" * 70)
    log(f"sanity start  {env_snapshot()}")

    dataset = PIEDataset(pie_root=Path(base["pie_root"]))
    log(f"dataset: {len(dataset)} samples, categories={dataset.category_counts()}")

    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"])
    info = model.describe()
    log(f"model: {info}")
    table = nu_table(model)
    for row in table:
        log(f"  t={row['t']} step={row['timestep']} a={row['a']} b={row['b']} nu={row['nu']}")

    results = {"model": info, "nu_table": table, "env": env_snapshot(),
               "precision": {"unet": base["unet_precision"], "vae": base["vae_precision"]}}

    log("S1: R must vanish when prompts match ...")
    results["S1"] = check_s1(model, dataset)
    log(f"  -> {results['S1']}")

    log("S2: determinism ...")
    results["S2"] = check_s2(model, dataset)
    log(f"  -> {results['S2']}")

    log("S3: fp16 vs fp32 residual ...")
    results["S3"] = check_s3(dataset, n=min(20, args.limit), device=args.device,
                             base_precision=base["unet_precision"])
    log(f"  -> {results['S3']}")

    log(f"S4: audit gate on {min(args.limit, 50)} images (lam=1 + prox, t=0.9) ...")
    metrics = PieMetrics(device=args.device)
    results["S4"] = check_s4(model, dataset, metrics, args, n=min(args.limit, 50))
    log(f"  -> pass={results['S4']['pass']} mean PSNR={results['S4']['mean_psnr_unedit']:.2f} "
        f"CLIP-E={results['S4']['mean_clip_edited']:.2f} recon={results['S4']['mean_recon_psnr']:.2f}")

    results["all_pass"] = all(results[k]["pass"] for k in ("S1", "S2", "S3", "S4"))
    write_json(ROOT / args.out, results)
    log(f"all_pass = {results['all_pass']}   -> {args.out}")

    if not results["S3"]["pass"]:
        log("S3 FAILED: switch unet/vae precision to fp32 in configs/base.yaml and record in "
            "report/DEVIATIONS.md")
    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
