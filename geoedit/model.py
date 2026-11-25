"""SD-Turbo wrapper: noise schedule, VAE, prompt embeddings, x0 prediction.

Everything the experiments need from the diffusion model lives here, so that
``edit_ops.py`` / ``masks.py`` never touch diffusers directly.

Conventions (shared by every script)
------------------------------------
``t`` is a *normalised* time in ``(0, 1)``; the UNet timestep is
``tau = round(t * (num_train_timesteps - 1))``.  With ``abar = alphas_cumprod[tau]``::

    a = sqrt(abar),  b = sqrt(1 - abar),  z = a * x + b * eps,  nu = b^2 / a^2

``x0_pred`` converts whatever the scheduler predicts into ``x0``:
``epsilon`` -> ``(z - b * eps_hat) / a``;  ``v_prediction`` -> ``a * z - b * v_hat``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL_ROOT = Path("/data/lx/Model/SD-Turbo")
PRECISIONS = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


class SDTurbo:
    """Loads sd-turbo once; every method is batched and ``torch.no_grad``."""

    def __init__(
        self,
        model_root: Union[str, Path] = DEFAULT_MODEL_ROOT,
        device: str = "cuda:0",
        unet_precision: str = "fp16",
        vae_precision: str = "fp16",
        text_precision: str = "fp32",
        image_size: int = 512,
        max_length: int = 77,
        vae_only: bool = False,
    ) -> None:
        """``vae_only=True`` skips the UNet and the text encoder (5 GB of weights).

        Only ``encode`` / ``decode`` are usable then.  P3a needs nothing else: the
        residuals come from ``results/p3/a/R`` and the edit is pure latent arithmetic.
        """
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import AutoTokenizer, CLIPTextModel

        root = Path(model_root)
        scheduler_dir = root / "scheduler_ddpm"
        if not scheduler_dir.exists():
            raise FileNotFoundError(
                f"{scheduler_dir} missing. sd-turbo ships an EulerDiscreteScheduler while the "
                "one-step edit needs a DDPM alpha/beta table; generate it with "
                "scripts/prepare_sd_turbo.py"
            )

        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.max_length = int(max_length)
        self.unet_dtype = PRECISIONS[unet_precision]
        self.vae_dtype = PRECISIONS[vae_precision]
        self.text_dtype = PRECISIONS[text_precision]

        self.vae = AutoencoderKL.from_pretrained(
            str(root / "vae"), torch_dtype=self.vae_dtype
        ).to(self.device).eval()
        if vae_only:
            self.unet = None
            self.tokenizer = None
            self.text_encoder = None
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                str(root / "unet"), torch_dtype=self.unet_dtype
            ).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
            self.text_encoder = CLIPTextModel.from_pretrained(
                str(root / "text_encoder"), torch_dtype=self.text_dtype
            ).to(self.device).eval()
        self.scheduler = DDPMScheduler.from_pretrained(str(scheduler_dir))

        self.prediction_type = self.scheduler.config.prediction_type
        self.num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        self.max_timestep = self.num_train_timesteps - 1
        self.scaling_factor = float(self.vae.config.scaling_factor)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(torch.float32).to(self.device)

        self._embed_cache: Dict[str, torch.Tensor] = {}
        self.nfe = 0  # UNet forward counter (T-E3 / NFE reporting)

        for module in (self.unet, self.vae, self.text_encoder):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # Noise schedule
    # ------------------------------------------------------------------ #
    def timestep_index(self, t: float) -> int:
        tau = int(round(float(t) * self.max_timestep))
        return max(0, min(self.max_timestep, tau))

    def ab(self, t: float) -> tuple[float, float]:
        tau = self.timestep_index(t)
        abar = float(self.alphas_cumprod[tau].item())
        return float(np.sqrt(abar)), float(np.sqrt(max(0.0, 1.0 - abar)))

    def nu(self, t: float) -> float:
        a, b = self.ab(t)
        return (b / a) ** 2

    def ab_tensor(self, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        abar = self.alphas_cumprod[timesteps].to(torch.float32)
        a = abar.sqrt().view(-1, 1, 1, 1)
        b = (1.0 - abar).clamp_min(0).sqrt().view(-1, 1, 1, 1)
        return a, b

    # ------------------------------------------------------------------ #
    # Text
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def embed(self, prompts: Sequence[str]) -> torch.Tensor:
        """``[B, 77, 1024]`` truncated CLIP text embeddings in the UNet dtype."""
        if self.text_encoder is None:
            raise RuntimeError("embed() needs the text encoder; this SDTurbo was built with vae_only=True")
        missing = [p for p in prompts if p not in self._embed_cache]
        if missing:
            tokens = self.tokenizer(
                missing,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            out = self.text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]
            for prompt, row in zip(missing, out):
                self._embed_cache[prompt] = row.detach().clone()
        rows = torch.stack([self._embed_cache[p] for p in prompts], dim=0)
        return rows.to(device=self.device, dtype=self.unet_dtype)

    # ------------------------------------------------------------------ #
    # VAE
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """PIL images -> scaled latents ``[B, 4, 64, 64]``."""
        batch = torch.stack([self._to_tensor(image) for image in images], dim=0).to(self.device)
        batch = batch.to(self.vae_dtype)
        latents = self.vae.encode(batch).latent_dist.mode() * self.scaling_factor
        return latents.to(self.device)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> List[np.ndarray]:
        """Scaled latents -> list of ``uint8 [512, 512, 3]`` arrays."""
        latents = (latents / self.scaling_factor).to(dtype=self.vae_dtype, device=self.device)
        pixels = self.vae.decode(latents).sample
        pixels = ((pixels.float().clamp(-1, 1) + 1) / 2 * 255.0).round().to(torch.uint8)
        return [p.cpu().permute(1, 2, 0).numpy() for p in pixels]

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1)

    # ------------------------------------------------------------------ #
    # One-step denoiser
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def x0_pred(
        self,
        z: torch.Tensor,
        t: Union[float, torch.Tensor],
        emb: torch.Tensor,
    ) -> torch.Tensor:
        """``E[x0 | z, c]`` for a batch; ``t`` may be a scalar or one value per item."""
        if self.unet is None:
            raise RuntimeError("x0_pred() needs the UNet; this SDTurbo was built with vae_only=True")
        z = z.to(device=self.device, dtype=self.unet_dtype)
        emb = emb.to(device=self.device, dtype=self.unet_dtype)
        if isinstance(t, torch.Tensor):
            timesteps = (t.to(self.device, torch.float32) * self.max_timestep).round().long()
            timesteps = timesteps.clamp(0, self.max_timestep)
        else:
            timesteps = torch.full((z.shape[0],), self.timestep_index(t), dtype=torch.long,
                                   device=self.device)

        out = self.unet(z, timesteps, encoder_hidden_states=emb, return_dict=False)[0]
        self.nfe += 1  # one network evaluation, regardless of batch size

        a, b = self.ab_tensor(timesteps)
        a = a.to(z.dtype)
        b = b.to(z.dtype)
        if self.prediction_type == "epsilon":
            x0 = (z - b * out) / a
        elif self.prediction_type == "v_prediction":
            x0 = a * z - b * out
        elif self.prediction_type == "sample":
            x0 = out
        else:
            raise ValueError(f"unsupported prediction_type {self.prediction_type!r}")
        return x0

    @torch.no_grad()
    def eps_pred(
        self,
        z: torch.Tensor,
        t: Union[float, torch.Tensor],
        emb: torch.Tensor,
    ) -> torch.Tensor:
        """``E[eps | z, c]`` (the noise prediction, not x0).  One UNet forward."""
        if self.unet is None:
            raise RuntimeError("eps_pred() needs the UNet; this SDTurbo was built with vae_only=True")
        z = z.to(device=self.device, dtype=self.unet_dtype)
        emb = emb.to(device=self.device, dtype=self.unet_dtype)
        if isinstance(t, torch.Tensor):
            timesteps = (t.to(self.device, torch.float32) * self.max_timestep).round().long()
            timesteps = timesteps.clamp(0, self.max_timestep)
        else:
            timesteps = torch.full((z.shape[0],), self.timestep_index(t), dtype=torch.long,
                                   device=self.device)

        out = self.unet(z, timesteps, encoder_hidden_states=emb, return_dict=False)[0]
        self.nfe += 1

        a, b = self.ab_tensor(timesteps)
        a = a.to(z.dtype)
        b = b.to(z.dtype)
        if self.prediction_type == "epsilon":
            eps = out
        elif self.prediction_type == "v_prediction":
            # z = a x0 + b eps, v = a eps - b x0  =>  eps = a v + b z
            eps = a * out + b * z
        elif self.prediction_type == "sample":
            eps = (z - a * out) / b.clamp_min(1e-8)
        else:
            raise ValueError(f"unsupported prediction_type {self.prediction_type!r}")
        return eps

    # ------------------------------------------------------------------ #
    def reset_nfe(self) -> None:
        self.nfe = 0

    def describe(self) -> Dict[str, object]:
        a, b = self.ab(0.9)
        return {
            "prediction_type": self.prediction_type,
            "num_train_timesteps": self.num_train_timesteps,
            "scaling_factor": self.scaling_factor,
            "unet_dtype": str(self.unet_dtype),
            "vae_dtype": str(self.vae_dtype),
            "text_dtype": str(self.text_dtype),
            "a_t0.9": a,
            "b_t0.9": b,
        }
