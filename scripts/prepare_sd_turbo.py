"""Prepare a local ``sd-turbo`` checkout for ChordEdit and verify every component loads.

Two things this fixes:

1. ChordEdit's ``from_local_weights`` loads the scheduler with
   ``DDPMScheduler.from_pretrained``, but ``stabilityai/sd-turbo`` ships an
   ``EulerDiscreteScheduler``.  Loading the Euler config as a DDPM scheduler raises a
   ``_class_name`` mismatch, so we write a sibling ``scheduler_ddpm/`` that keeps the
   SD2.1 beta schedule (``scaled_linear``, 0.00085 -> 0.012, 1000 steps) and the
   epsilon parametrisation, while being loadable as ``DDPMScheduler``.

2. It validates the layout and prints parameter counts so a truncated download shows up
   immediately instead of as a cryptic shape error 700 images into a run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = [
    Path("/data/lx/Model/SD-Turbo"),
    Path("/data/lx/Model/sd-turbo"),
    Path("/data/lx/Model/sd_turbo"),
]

DDPM_CONFIG = {
    "_class_name": "DDPMScheduler",
    "_diffusers_version": "0.31.0",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "rescale_betas_zero_snr": False,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "trained_betas": None,
    "variance_type": "fixed_small",
}


def find_model_root(explicit: str | None = None) -> Path:
    if explicit:
        root = Path(explicit)
        if not root.exists():
            raise FileNotFoundError(root)
        return root
    for candidate in CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"none of {CANDIDATES} exists; pass --model-root")


def component_paths(model_root: Path, scheduler_dir: str = "scheduler_ddpm") -> Dict[str, str]:
    paths = {
        "unet_path": model_root / "unet",
        "scheduler_path": model_root / scheduler_dir,
        "vae_path": model_root / "vae",
        "tokenizer_path": model_root / "tokenizer",
        "text_encoder_path": model_root / "text_encoder",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing model components:\n  " + "\n  ".join(missing))
    return {k: str(v) for k, v in paths.items()}


def ensure_ddpm_scheduler(model_root: Path) -> Path:
    out_dir = model_root / "scheduler_ddpm"
    out_dir.mkdir(exist_ok=True)
    config_path = out_dir / "scheduler_config.json"
    config_path.write_text(json.dumps(DDPM_CONFIG, indent=2) + "\n")
    return out_dir


def validate(paths: Dict[str, str], device: str = "cpu") -> None:
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import AutoTokenizer, CLIPTextModel

    unet = UNet2DConditionModel.from_pretrained(paths["unet_path"], torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(paths["vae_path"], torch_dtype=torch.float32)
    scheduler = DDPMScheduler.from_pretrained(paths["scheduler_path"])
    text_encoder = CLIPTextModel.from_pretrained(paths["text_encoder_path"])
    tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer_path"])

    print(f"  unet          : {sum(p.numel() for p in unet.parameters()) / 1e6:8.1f} M params")
    print(f"  vae           : {sum(p.numel() for p in vae.parameters()) / 1e6:8.1f} M params, "
          f"scaling_factor={vae.config.scaling_factor}")
    print(f"  text_encoder  : {sum(p.numel() for p in text_encoder.parameters()) / 1e6:8.1f} M params, "
          f"hidden={text_encoder.config.hidden_size}")
    print(f"  tokenizer     : vocab={tokenizer.vocab_size}")
    print(f"  scheduler     : {type(scheduler).__name__}, "
          f"timesteps={scheduler.config.num_train_timesteps}, "
          f"betas per schedule={scheduler.betas.shape[0]}")

    ac = scheduler.alphas_cumprod
    for t_norm in (0.15, 0.30, 0.75, 0.90):
        step = round((scheduler.config.num_train_timesteps - 1) * t_norm)
        a = float(ac[step].sqrt())
        s = float((1 - ac[step]).sqrt())
        print(f"    t={t_norm:.2f} -> step {step:4d}  alpha={a:.4f} sigma={s:.4f}")

    # One tiny VAE round trip: catches a truncated / mismatched safetensors early.
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode() * vae.config.scaling_factor
        y = vae.decode(z / vae.config.scaling_factor).sample
    print(f"  vae round trip: {tuple(x.shape)} -> {tuple(z.shape)} -> {tuple(y.shape)}  OK")
    del unet, vae, text_encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    root = find_model_root(args.model_root)
    print(f"model root: {root}")
    print(f"writing DDPM scheduler config -> {root / 'scheduler_ddpm' / 'scheduler_config.json'}")
    ensure_ddpm_scheduler(root)
    paths = component_paths(root)
    for key, value in paths.items():
        print(f"  {key:18s} {value}")
    print("validating components ...")
    validate(paths, device=args.device)
    print("OK: sd-turbo layout is usable")


if __name__ == "__main__":
    sys.exit(main())
