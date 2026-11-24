"""Thin wrapper around the *official* ChordEdit pipeline.

We deliberately do not reimplement the transport: ``third_party/ChordEdit`` is pinned
(see ``third_party/PINS.md``) and everything below calls its private methods in the same
order as ``ChordEditPipeline.__call__`` so that our cached proposals are bit-identical to
a plain ``python run_pie_bench.py`` run.

The only thing we add is the ability to stop after the transport step (``cleanup=False``)
and to run the proximal step batched over many images, which is where all the cost of the
OT experiments lives (one extra UNet call per image per config).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CHORD_REPO = ROOT / "third_party" / "ChordEdit"
if str(CHORD_REPO) not in sys.path:
    sys.path.insert(0, str(CHORD_REPO))

from pipeline_chord import ChordEditPipeline  # noqa: E402  (needs the sys.path tweak)

DEFAULT_MODEL_ROOT = Path("/data/lx/Model/SD-Turbo")

# Same defaults as third_party/ChordEdit/run_pie_bench.py, which is what the paper reports.
DEFAULT_EDIT_CONFIG: Dict[str, float] = {
    "noise_samples": 1,
    "n_steps": 1,
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.15,
    "step_scale": 1.0,
    "cleanup": False,  # transport only; the prox is applied explicitly
}
DEFAULT_SEED = 42


def component_paths(model_root: Path = DEFAULT_MODEL_ROOT) -> Dict[str, str]:
    return {
        "unet_path": str(model_root / "unet"),
        "scheduler_path": str(model_root / "scheduler_ddpm"),
        "vae_path": str(model_root / "vae"),
        "tokenizer_path": str(model_root / "tokenizer"),
        "text_encoder_path": str(model_root / "text_encoder"),
    }


class ChordEditor:
    """Transport + proximal step + VAE decode, on one device."""

    def __init__(
        self,
        model_root: Path = DEFAULT_MODEL_ROOT,
        device: str = "cuda:0",
        seed: int = DEFAULT_SEED,
        image_size: int = 512,
        edit_config: Optional[Dict[str, float]] = None,
    ) -> None:
        self.device = torch.device(device)
        self.seed = int(seed)
        self.image_size = int(image_size)
        self.edit_config = dict(DEFAULT_EDIT_CONFIG)
        if edit_config:
            self.edit_config.update(edit_config)

        paths = component_paths(model_root)
        for key, value in paths.items():
            if not Path(value).exists():
                raise FileNotFoundError(f"{key} does not exist: {value}")
        self.pipe = ChordEditPipeline.from_local_weights(
            paths,
            default_edit_config=self.edit_config,
            device=str(self.device),
            torch_dtype=torch.float32,
            image_size=self.image_size,
            use_center_crop=True,
            use_safety_checker=False,
        )
        self.pipe.to(self.device)
        self.pipe.unet.eval()
        self.pipe.vae.eval()
        self.pipe.text_encoder.eval()
        self._embed_cache: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def pixel(self, image: Image.Image) -> torch.Tensor:
        return self.pipe._prepare_image_tensor(image)

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """PIL -> ``x_src`` in *scaled* latent units, ``[1, 4, 64, 64]``."""
        return self.pipe._encode_image_to_latent(self.pipe._prepare_image_tensor(image))

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        cached = self._embed_cache.get(prompt)
        if cached is None:
            cached = self.pipe.encode_prompt([prompt]).to(self.device)
            self._embed_cache[prompt] = cached
        return cached

    @torch.no_grad()
    def transport(
        self,
        image: Image.Image,
        source_prompt: str,
        target_prompt: str,
        seed: Optional[int] = None,
    ) -> Dict[str, object]:
        """One ChordEdit transport step (``x_src + step_scale * u_hat``), no proximal step.

        Returns ``x_src``, ``y``, the sampler ``noise`` (needed to reproduce both the
        transport and the prox exactly) and the wall time.
        """
        seed = self.seed if seed is None else int(seed)
        t0 = time.perf_counter()

        x_src = self.encode(image)
        src_embed = self.encode_prompt(source_prompt)
        tgt_embed = self.encode_prompt(target_prompt)
        params = self.pipe._prepare_edit_params(dict(self.edit_config))
        noise = self.pipe._prepare_noise_list(x_src, seed, 1)[0]
        y = self.pipe._run_edit(x_src, src_embed, tgt_embed, [noise], params)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "x_src": x_src.detach(),
            "y": y.detach(),
            "noise": noise.detach(),
            "seed": seed,
            "seconds": time.perf_counter() - t0,
        }

    # ------------------------------------------------------------------ #
    # Proximal step and decode
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def prox(
        self,
        latents: torch.Tensor,
        conds: torch.Tensor,
        noises: torch.Tensor,
        t_c: Optional[float] = None,
    ) -> torch.Tensor:
        """Batched ``prox(x, t_c, c_tar) = xhat0(z_t)``, matching ``_pred_x0`` exactly.

        ``latents`` / ``noises``: ``[B, 4, 64, 64]``; ``conds``: ``[B, 77, 1024]``.
        """
        t_c = self.edit_config["t_end"] if t_c is None else float(t_c)
        timesteps = self.pipe._time_to_index(latents.shape[0], t_c, device=latents.device)
        alpha, sigma = self.pipe._get_alpha_sigma(latents, timesteps)
        z = alpha * latents + sigma * noises
        eps = self.pipe.unet(
            sample=z, timestep=timesteps, encoder_hidden_states=conds, return_dict=False
        )[0]
        return (z - sigma * eps) / alpha

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> List[np.ndarray]:
        """``[B, 4, h, w]`` scaled latents -> list of ``uint8`` ``[512, 512, 3]`` arrays."""
        images = self.pipe._decode_latent_to_image(latents)
        array = (images.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1)
        return [a.cpu().numpy() for a in array]
