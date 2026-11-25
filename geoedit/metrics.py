"""PIE-Bench metrics, delegating to the *official* PnPInversion implementations.

``third_party/PnPInversion/evaluation/matrics_calculator.py`` is used unmodified.  The
only local adaptation is that its hard-coded ``openai/clip-vit-large-patch14`` is
redirected to the local copy at ``/data/lx/Model/CLIP`` (there is no direct HF access on
this machine, and the repo id would otherwise trigger a download).  MetricsCalculator
imports ``CLIPScore`` at module load, so the patch has to happen before that import.

Reported columns follow the official PIE-Bench protocol:

===============  ==========================================================
``psnr`` etc.    whole image
``*_unedit``     background only (``1 - mask``), the numbers papers report
``clip_whole``   output vs **source** prompt
``clip_edited``  output vs **target** prompt
``struct_dist``  DINO ViT-B/8 structure distance
===============  ==========================================================
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PNP_REPO = ROOT / "third_party" / "PnPInversion"

# torchmetrics only accepts a path containing "openai" (it branches on that substring),
# so we expose the local CLIP copy through a namespaced symlink inside the repo cache.
CLIP_SOURCE = Path("/data/lx/Model/CLIP")
LOCAL_CLIP = ROOT / "cache" / "models" / "openai" / "clip-vit-large-patch14"

# Keep torch.hub / HF caches on /data instead of the root disk.
os.environ.setdefault("TORCH_HOME", "/data/lx/.cache/torch")
os.environ.setdefault("HF_HOME", "/data/lx/.cache/huggingface")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_METRIC_KEYS = (
    "psnr",
    "lpips",
    "mse",
    "ssim",
    "psnr_unedit",
    "lpips_unedit",
    "mse_unedit",
    "ssim_unedit",
    "clip_whole",
    "clip_edited",
    "clip_edited_edit_part",
    "struct_dist",
)


def _ensure_clip_symlink() -> None:
    """Make ``cache/models/openai/clip-vit-large-patch14`` point at a local CLIP copy."""
    if LOCAL_CLIP.exists():
        return
    if not CLIP_SOURCE.exists():
        raise FileNotFoundError(
            f"neither {LOCAL_CLIP} nor {CLIP_SOURCE} exists; PnPInversion needs CLIP ViT-L/14"
        )
    LOCAL_CLIP.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_CLIP.symlink_to(CLIP_SOURCE)


def _load_calculator():
    _ensure_clip_symlink()
    for path in (str(PNP_REPO), str(PNP_REPO / "evaluation")):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torchmetrics.multimodal as torchmetrics_multimodal

    if not getattr(torchmetrics_multimodal, "_geoedit_patched", False):
        original = torchmetrics_multimodal.CLIPScore

        class _LocalCLIPScore(original):  # type: ignore[misc, valid-type]
            def __init__(self, model_name_or_path=None, **kwargs):
                super().__init__(model_name_or_path=str(LOCAL_CLIP), **kwargs)

        torchmetrics_multimodal.CLIPScore = _LocalCLIPScore
        torchmetrics_multimodal._geoedit_patched = True

    from evaluation.matrics_calculator import MetricsCalculator  # noqa: E402

    return MetricsCalculator


class PieMetrics:
    """One instance per process; holds CLIP / LPIPS / DINO weights."""

    def __init__(self, device: str = "cuda:0") -> None:
        calculator_cls = _load_calculator()
        self.device = torch.device(device)
        self.calc = calculator_cls(device=str(self.device))

    def evaluate(
        self,
        source: np.ndarray,
        output: np.ndarray,
        mask: np.ndarray,
        source_prompt: str,
        target_prompt: str,
        lean: bool = False,
    ) -> Dict[str, float]:
        """``source`` / ``output``: ``uint8 [512, 512, 3]``; ``mask``: ``[512, 512]`` 0/1."""
        calc = self.calc
        mask = np.asarray(mask).astype(np.float32)
        if mask.ndim == 2:
            # The official evaluator repeats the mask over RGB before masking images.
            mask = mask[:, :, np.newaxis].repeat(3, axis=2)
        background = 1.0 - mask

        out: Dict[str, float] = {}
        # lean=True keeps only the metrics §4.2 of the v2 plan asks for; the whole-image
        # variants are not deliverables and each costs a full forward pass.
        if not lean:
            out["psnr"] = float(calc.calculate_psnr(source, output, None, None))
            out["lpips"] = float(calc.calculate_lpips(source, output, None, None))
            out["mse"] = float(calc.calculate_mse(source, output, None, None))
            out["ssim"] = float(calc.calculate_ssim(source, output, None, None))

        # The official evaluator returns "nan" when the background is empty (style edits whose
        # mask covers the frame).  Without this guard PSNR becomes +inf and poisons means.
        has_background = bool(background.sum() > 0)
        nan = float("nan")
        out["psnr_unedit"] = float(calc.calculate_psnr(source, output, background, background)) if has_background else nan
        out["lpips_unedit"] = float(calc.calculate_lpips(source, output, background, background)) if has_background else nan
        out["mse_unedit"] = float(calc.calculate_mse(source, output, background, background)) if has_background else nan
        out["ssim_unedit"] = float(calc.calculate_ssim(source, output, background, background)) if has_background else nan
        out["has_background"] = float(has_background)

        out["clip_whole"] = float(calc.calculate_clip_similarity(output, source_prompt, None))
        out["clip_edited"] = float(calc.calculate_clip_similarity(output, target_prompt, None))
        if not lean:
            out["clip_edited_edit_part"] = float(
                calc.calculate_clip_similarity(output, target_prompt, mask)
            )
        out["struct_dist"] = float(calc.calculate_structure_distance(source, output, None, None))
        return out
