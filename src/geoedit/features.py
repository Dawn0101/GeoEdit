"""Token features for the OT cost, on a common 64x64 grid.

Three options, matching task document section 6.3 but with the grid unified to the latent
resolution (64x64 for 512px inputs) instead of 32x32:

``lat``   latent (4 ch) with a 3x3 unfold -> 36-d per cell.  Colour sensitive, fine grid.
``dino``  DINOv2 ViT patch features, bilinearly upsampled from 32x32 (448/14) to 64x64.
``both``  the two blocks concatenated, each divided by its own calibration scale so that
          ``both`` has the same units as either single block.

Why 64x64 everywhere: ``sigma`` is used to blend latents, so a 32x32 mask is upsampled
anyway and its boundary uncertainty is 16 latent cells.  Solving on the latent grid keeps
``sigma`` aligned with the pixels it acts on and removes one interpolation.

Scaling convention: each block is divided by its own ``s_f`` (median same-position drift)
*before* concatenation, so a block whose features simply have a larger norm does not
dominate the cost.  ``estimate_sigma`` then re-calibrates ``s_f`` on the concatenated
vector, which is what keeps ``tau`` comparable across ``feat`` settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_DINOV2 = Path("/data/lx/Model/DINOv2-Large")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def latent_unfold(x_latent: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """``[B, 4, h, w]`` -> ``[B, N, 4*k*k]`` with replicate padding at the border."""
    if x_latent.dim() == 3:
        x_latent = x_latent[None]
    pad = kernel // 2
    padded = F.pad(x_latent, (pad,) * 4, mode="replicate")
    patches = F.unfold(padded, kernel_size=kernel, stride=1)  # [B, 4*k*k, N]
    return patches.transpose(1, 2).contiguous()


class DinoExtractor:
    """DINOv2 patch tokens at 448 -> 32x32, upsampled to the OT grid."""

    def __init__(self, model_path: Path = DEFAULT_DINOV2, device: str = "cuda:0",
                 input_size: int = 448) -> None:
        from transformers import Dinov2Model

        self.device = torch.device(device)
        self.input_size = int(input_size)
        self.model = Dinov2Model.from_pretrained(str(model_path)).to(self.device).eval()
        self.patch = int(self.model.config.patch_size)
        self.grid = self.input_size // self.patch
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def __call__(self, images: Sequence[np.ndarray], grid: int) -> torch.Tensor:
        """``images``: list of ``uint8 [H, W, 3]``.  Returns ``[B, grid*grid, D]``."""
        batch = torch.stack([self._to_tensor(image) for image in images]).to(self.device)
        tokens = self.model(batch, interpolate_pos_encoding=True).last_hidden_state
        tokens = tokens[:, 1:]  # drop the CLS token
        tokens = F.normalize(tokens, dim=-1)  # L2, so the cost is a cosine distance
        tokens = tokens.reshape(len(images), self.grid, self.grid, self.dim).permute(0, 3, 1, 2)
        if self.grid != grid:
            tokens = F.interpolate(tokens, size=(grid, grid), mode="bilinear", align_corners=False)
        return tokens.permute(0, 2, 3, 1).reshape(len(images), grid * grid, self.dim)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1) / 255.0
        x = F.interpolate(x[None], size=(self.input_size, self.input_size),
                          mode="bicubic", align_corners=False, antialias=True)[0]
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (x - mean) / std


def block_scale(feat_src: torch.Tensor, feat_tgt: torch.Tensor, q: float = 0.5) -> torch.Tensor:
    """Median same-position distance for one block, the unit in which ``tau`` is read."""
    d = ((feat_src - feat_tgt) ** 2).sum(-1)
    return torch.quantile(d, q).clamp_min(1e-8).sqrt()


@dataclass
class FeatureBundle:
    """Features for one image pair, plus the bookkeeping E1/E4 need."""

    src: torch.Tensor  # [N, D]
    tgt: torch.Tensor  # [N, D]
    grid: int
    kind: str
    block_scales: Dict[str, float] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.src.shape[1])


class EditFeatureExtractor:
    """Builds the OT feature pair ``(f_src, f_proposal)`` for a given ``kind``."""

    def __init__(self, kind: str = "lat", device: str = "cuda:0", grid: int = 64,
                 unfold: int = 3, dino_path: Path = DEFAULT_DINOV2) -> None:
        if kind not in ("lat", "dino", "both"):
            raise ValueError(f"unknown feature kind {kind!r}")
        self.kind = kind
        self.device = torch.device(device)
        self.grid = int(grid)
        self.unfold = int(unfold)
        self._dino: Optional[DinoExtractor] = None
        self.dino_path = dino_path

    @property
    def dino(self) -> DinoExtractor:
        if self._dino is None:
            self._dino = DinoExtractor(self.dino_path, str(self.device))
        return self._dino

    def _blocks(self, x_src: torch.Tensor, x_tgt: torch.Tensor,
                image_src: Optional[np.ndarray], image_tgt: Optional[np.ndarray],
                frame: Dict[str, Dict[str, torch.Tensor]]) -> List[tuple]:
        """Return ``[(name, f_src, f_tgt), ...]`` for the requested kind."""
        blocks = []
        if self.kind in ("lat", "both"):
            blocks.append((
                "lat",
                latent_unfold(x_src, self.unfold),
                latent_unfold(x_tgt, self.unfold),
            ))
        if self.kind in ("dino", "both"):
            if frame.get("dino_tgt") is None:
                raise ValueError("dino features require the decoded images (src and proposal)")
            blocks.append(("dino", frame["dino_src"], frame["dino_tgt"]))
        return blocks

    def __call__(
        self,
        x_src: torch.Tensor,
        x_tgt: torch.Tensor,
        image_src: Optional[np.ndarray] = None,
        image_tgt: Optional[np.ndarray] = None,
    ) -> FeatureBundle:
        """``x_*``: ``[1, 4, 64, 64]`` scaled latents; ``image_*``: ``uint8 [512, 512, 3]``."""
        unbatched = x_src.dim() == 3
        if unbatched:
            x_src = x_src[None]
            x_tgt = x_tgt[None]
        x_src = x_src.to(self.device)
        x_tgt = x_tgt.to(self.device)
        frame: Dict[str, Optional[torch.Tensor]] = {"dino_src": None, "dino_tgt": None}
        if self.kind in ("dino", "both"):
            if image_src is None or image_tgt is None:
                raise ValueError("dino features require decoded images")
            out = self.dino([image_src, image_tgt], self.grid)
            frame["dino_src"], frame["dino_tgt"] = out[0], out[1]

        blocks = self._blocks(x_src, x_tgt, image_src, image_tgt, frame)
        scales: Dict[str, float] = {}
        parts_src, parts_tgt = [], []
        for name, f_src, f_tgt in blocks:
            scale = block_scale(f_src, f_tgt)
            scales[name] = float(scale)
            parts_src.append(f_src / scale)
            parts_tgt.append(f_tgt / scale)

        n_blocks = len(parts_src)
        src = torch.cat(parts_src, dim=-1) / n_blocks
        tgt = torch.cat(parts_tgt, dim=-1) / n_blocks
        if unbatched:
            src, tgt = src[0], tgt[0]
        return FeatureBundle(src=src, tgt=tgt, grid=self.grid, kind=self.kind,
                             block_scales=scales)


def dino_patch_features(
    images: Sequence[np.ndarray],
    model_path: Path = DEFAULT_DINOV2,
    device: str = "cuda:0",
    input_size: int = 448,
) -> torch.Tensor:
    """Convenience wrapper used by E1 (patch cosine distances on decoded images)."""
    extractor = DinoExtractor(model_path, device=device, input_size=input_size)
    grid = input_size // extractor.patch
    return extractor(images, grid)
