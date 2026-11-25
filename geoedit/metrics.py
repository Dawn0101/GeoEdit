"""PIE-Bench metrics, delegating to the *official* PnPInversion implementations.

``third_party/PnPInversion/evaluation/matrics_calculator.py`` is used unmodified.  The only
local adaptation is that its hard-coded ``openai/clip-vit-large-patch14`` is redirected to
the local copy at ``/data/lx/Model/CLIP`` (there is no direct HF access on this machine, and
the repo id would otherwise trigger a download).  MetricsCalculator imports ``CLIPScore`` at
module load, so the patch has to happen before that import.

Column names follow the **official** ``evaluation/evaluate.py`` exactly, so a number can be
looked up in the paper's table without guessing:

====================================  =====================================================
``clip_source_whole``                 ``clip_similarity_source_image``   output vs **source** prompt, whole image
``clip_target_whole``                 ``clip_similarity_target_image``   output vs **target** prompt, whole image
``clip_target_edit``                  ``clip_similarity_target_image_edit_part``  output*mask vs target prompt
``psnr`` / ``lpips`` / ``mse`` / ``ssim``                          whole image
``*_unedit``                          background only (``1 - mask``) -- what papers call "All"/"PSNR"
``struct_dist``                       ``structure_distance`` (DINO ViT-B/8)
====================================  =====================================================

Naming bug fixed in v3 (D10): v2 called ``clip_source_whole`` "clip_whole" and
``clip_target_whole`` "clip_edited", which made the P2 report ambiguous.  The *values* were
correct; only the labels changed.  ``clip_target_edit`` used to be skipped whenever
``lean=True``; it is now always computed, because it is P3's primary semantic metric.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

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

#: every key ``evaluate()`` can return (``whole=True`` adds the four whole-image ones)
METRIC_KEYS = (
    "psnr_unedit",
    "lpips_unedit",
    "mse_unedit",
    "ssim_unedit",
    "struct_dist",
    "clip_source_whole",
    "clip_target_whole",
    "clip_target_edit",
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
        import torch

        self.device = torch.device(device)
        self.calc = calculator_cls(device=str(self.device))
        self._text_cache: Dict[str, object] = {}

    def evaluate(
        self,
        source: np.ndarray,
        output: np.ndarray,
        mask: np.ndarray,
        source_prompt: str,
        target_prompt: str,
        whole: bool = False,
    ) -> Dict[str, float]:
        """``source`` / ``output``: ``uint8 [512, 512, 3]``; ``mask``: ``[512, 512]`` 0/1."""
        calc = self.calc
        mask = np.asarray(mask).astype(np.float32)
        if mask.ndim == 2:
            # The official evaluator repeats the mask over RGB before masking images.
            mask = mask[:, :, np.newaxis].repeat(3, axis=2)
        background = 1.0 - mask

        out: Dict[str, float] = {}
        if whole:
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

        out["clip_source_whole"] = float(calc.calculate_clip_similarity(output, source_prompt, None))
        out["clip_target_whole"] = float(calc.calculate_clip_similarity(output, target_prompt, None))
        out["clip_target_edit"] = (
            float(calc.calculate_clip_similarity(output, target_prompt, mask))
            if bool((mask > 0).any()) else nan
        )
        out["struct_dist"] = float(calc.calculate_structure_distance(source, output, None, None))
        return out

    # ------------------------------------------------------------------ #
    # Batched variant (P3a evaluates ~163 latents per image)
    # ------------------------------------------------------------------ #
    def _clip_towers(self):
        metric = self.calc.clip_metric_calculator
        metric.model.to(self.device)
        return metric.model, metric.processor

    def _image_features(self, images: Sequence[np.ndarray]):
        """``[B, D]`` CLIP image features, using torchmetrics' own preprocessing."""
        import torch
        from torchmetrics.functional.multimodal.clip_score import _get_features

        model, processor = self._clip_towers()
        tensors = [torch.tensor(np.asarray(im)).permute(2, 0, 1) for im in images]
        with torch.no_grad():
            return _get_features(tensors, "image", self.device, model, processor)

    def _text_features(self, prompt: str):
        import torch
        from torchmetrics.functional.multimodal.clip_score import _get_features

        if prompt not in self._text_cache:
            model, processor = self._clip_towers()
            with torch.no_grad():
                self._text_cache[prompt] = _get_features(
                    [prompt], "text", self.device, model, processor).detach()
        return self._text_cache[prompt]

    @staticmethod
    def _cosine(image_features, text_features):
        """Same arithmetic as ``_clip_score_update``: 100 * cos(E_I, E_C), clamped at 0."""
        import torch

        a = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        b = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return torch.clamp(100.0 * (a * b).sum(dim=-1), min=0.0)

    def evaluate_many(
        self,
        source: np.ndarray,
        outputs: Sequence[np.ndarray],
        mask: np.ndarray,
        source_prompt: str,
        target_prompt: str,
        chunk: int = 12,
    ) -> List[Dict[str, float]]:
        """Same numbers as :meth:`evaluate`, but the three CLIP terms are batched.

        Unbatched, each output re-encodes the image *three* times (whole/source, whole/target,
        masked/target) and re-encodes the prompt each time; for 163 outputs per image that is
        ~80% of the evaluation time.  Here the image features are computed once per chunk (one
        call for the whole images, one for the masked ones) and the text features are cached
        for the whole run, so only independent forward passes are regrouped -- the numbers are
        identical up to float noise.  ``test_p3.py::test_t_x3b_batched_matches_single`` pins
        that down against :meth:`evaluate` (the official path).
        """
        mask3 = np.asarray(mask).astype(np.float32)
        if mask3.ndim == 2:
            mask3 = mask3[:, :, np.newaxis].repeat(3, axis=2)
        background = 1.0 - mask3
        has_background = bool(background.sum() > 0)
        has_edit = bool((mask3 > 0).any())
        nan = float("nan")
        src_text = self._text_features(source_prompt)
        tar_text = self._text_features(target_prompt)

        calc = self.calc
        results: List[Dict[str, float]] = []
        for start in range(0, len(outputs), int(chunk)):
            block = outputs[start:start + int(chunk)]
            whole_features = self._image_features(block)
            masked_features = (self._image_features([np.uint8(np.asarray(o) * mask3) for o in block])
                               if has_edit else None)
            clip_src = self._cosine(whole_features, src_text).float().cpu()
            clip_tar = self._cosine(whole_features, tar_text).float().cpu()
            clip_edit = (None if masked_features is None
                         else self._cosine(masked_features, tar_text).float().cpu())

            for k, output in enumerate(block):
                row: Dict[str, float] = {}
                row["psnr_unedit"] = float(calc.calculate_psnr(source, output, background, background)) if has_background else nan
                row["lpips_unedit"] = float(calc.calculate_lpips(source, output, background, background)) if has_background else nan
                row["mse_unedit"] = float(calc.calculate_mse(source, output, background, background)) if has_background else nan
                row["ssim_unedit"] = float(calc.calculate_ssim(source, output, background, background)) if has_background else nan
                row["has_background"] = float(has_background)
                row["clip_source_whole"] = float(clip_src[k])
                row["clip_target_whole"] = float(clip_tar[k])
                row["clip_target_edit"] = float(clip_edit[k]) if clip_edit is not None else nan
                row["struct_dist"] = float(calc.calculate_structure_distance(source, output, None, None))
                results.append(row)
        return results
