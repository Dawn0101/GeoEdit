"""PIE-Bench protocol helpers: mapping file, RLE masks, prompt cleanup.

Mirrors the official ``cure-lab/PnPInversion`` conventions so that numbers are directly
comparable with ChordEdit / PnPInversion tables:

* the ``[]`` markers around the edited word are stripped from the prompts;
* masks are decoded with the official ``mask_decode`` (row-major RLE of 0/1) and the
  outermost 1-pixel border is forced to 1 to absorb annotation noise;
* the background for metrics is ``1 - mask``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

DEFAULT_PIE_ROOT = Path("/data/lx/Dataset/PIE-Bench-Dataset")


@dataclass(frozen=True)
class PieSample:
    image_id: str
    image_path: Path
    original_prompt: str
    editing_prompt: str
    editing_instruction: str
    editing_type_id: str
    rle_mask: Tuple[int, ...]
    blended_word: str = ""  # "bicycle bicycle" -- the edited word in src / in tar


def strip_brackets(text: str) -> str:
    """``"a [cat] on a chair"`` -> ``"a cat on a chair"`` (PIE-Bench prompt cleanup)."""
    return re.sub(r"[\[\]]", "", text).strip()


def load_mapping(pie_root: Path = DEFAULT_PIE_ROOT) -> Dict[str, PieSample]:
    root = Path(pie_root)
    data = json.loads((root / "mapping_file.json").read_text())
    out: Dict[str, PieSample] = {}
    for image_id, item in data.items():
        out[image_id] = PieSample(
            image_id=image_id,
            image_path=root / "annotation_images" / item["image_path"],
            original_prompt=strip_brackets(item["original_prompt"]),
            editing_prompt=strip_brackets(item["editing_prompt"]),
            editing_instruction=item.get("editing_instruction", ""),
            editing_type_id=str(item["editing_type_id"]),
            rle_mask=tuple(item["mask"]),
            blended_word=item.get("blended_word", ""),
        )
    return out


def mask_decode(encoded_mask, image_shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Official PnPInversion decoder (kept identical, including the border fix-up)."""
    height, width = image_shape
    length = height * width
    mask_array = np.zeros((length,), dtype=np.uint8)
    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1
    mask_array = mask_array.reshape(height, width)
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1
    return mask_array


def to_latent_grid(mask: np.ndarray, size: int = 64, erode: int = 2) -> torch.Tensor:
    """``[H, W]`` 0/1 mask -> ``[1, 1, size, size]`` float mask, optionally eroded.

    ``erode > 0`` shrinks the edit mask in *latent* cells, which is how E1 defines the
    background (the mask complement eroded by 2 latent cells = 16 image pixels).
    """
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    if m.shape[-1] != size:
        m = torch.nn.functional.interpolate(m, size=(size, size), mode="nearest")
    if erode > 0:
        k = 2 * erode + 1
        pad = torch.nn.functional.pad(m, (erode,) * 4, mode="replicate")
        m = -torch.nn.functional.max_pool2d(-pad, kernel_size=k, stride=1)
    return m.clamp(0, 1)


def edit_area_fraction(mask: np.ndarray) -> float:
    return float(np.asarray(mask).mean())


def select_samples(
    mapping: Dict[str, PieSample],
    per_type: int = 1,
    types: Optional[List[str]] = None,
) -> List[PieSample]:
    """Deterministic, evenly spread sample selection (one or more per edit type)."""
    by_type: Dict[str, List[PieSample]] = {}
    for sample in mapping.values():
        by_type.setdefault(sample.editing_type_id, []).append(sample)
    chosen: List[PieSample] = []
    for type_id in sorted(by_type, key=lambda t: int(t)):
        if types is not None and type_id not in types:
            continue
        items = sorted(by_type[type_id], key=lambda s: s.image_id)
        if per_type >= len(items):
            chosen.extend(items)
        else:
            step = len(items) / per_type
            chosen.extend(items[int(i * step)] for i in range(per_type))
    return chosen
