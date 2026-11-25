"""PIE-Bench dataset: prompts, images, GT masks, latent masks and distance maps.

The mask handling mirrors the official PnPInversion evaluator (same RLE decoding and the
same 1-pixel border fix-up), so ``mask512`` is directly comparable with the published
numbers.  ``mask64`` / ``dist64`` are what the halo statistics need: an 8x8 average-pooled
interior mask on the latent grid, and the Euclidean distance from every latent cell to the
nearest interior cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
from PIL import Image
from scipy import ndimage

from .pie import DEFAULT_PIE_ROOT, PieSample, load_mapping, mask_decode

LATENT = 64
POOL = 8  # 512 / 64


@dataclass
class PIESample:
    image_id: str
    category: str
    image_path: Path
    src_prompt: str
    tar_prompt: str
    mask512: np.ndarray  # bool [512, 512]
    mask64: np.ndarray  # float32 [64, 64] in {0, 1}
    dist64: np.ndarray  # float32 [64, 64], distance to nearest interior latent cell

    @property
    def interior_fraction(self) -> float:
        return float(self.mask64.mean())

    def image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")


def build_mask64(mask512: np.ndarray) -> np.ndarray:
    """8x8 average pool of the 512^2 mask, then threshold at 0.5 -> {0,1} on 64x64."""
    pooled = mask512.astype(np.float32).reshape(LATENT, POOL, LATENT, POOL).mean(axis=(1, 3))
    return (pooled > 0.5).astype(np.float32)


def build_dist64(mask64: np.ndarray) -> np.ndarray:
    """Euclidean distance (in latent cells) from each cell to the nearest interior cell."""
    interior = mask64 > 0.5
    if not interior.any():
        return np.full_like(mask64, np.inf, dtype=np.float32)
    # distance_transform_edt measures distance to the nearest *zero* entry, so pass ~I.
    return ndimage.distance_transform_edt(~interior).astype(np.float32)


class PIEDataset:
    """All 700 PIE-Bench samples with a persistent per-image cache."""

    def __init__(
        self,
        pie_root: Path = DEFAULT_PIE_ROOT,
        cache_dir: Optional[Path] = Path("cache/pie"),
    ) -> None:
        self.pie_root = Path(pie_root)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mapping: Dict[str, PieSample] = load_mapping(self.pie_root)
        self.image_ids: List[str] = sorted(self._mapping)
        self._cache: Dict[str, PIESample] = {}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __iter__(self) -> Iterator[PIESample]:
        for image_id in self.image_ids:
            yield self[image_id]

    def __getitem__(self, index) -> PIESample:
        image_id = self.image_ids[index] if isinstance(index, int) else str(index)
        if image_id in self._cache:
            return self._cache[image_id]
        base = self._mapping[image_id]
        entry = None
        if self.cache_dir is not None:
            path = self.cache_dir / f"{image_id}.npz"
            if path.exists():
                with np.load(path) as data:
                    entry = PIESample(
                        image_id=image_id,
                        category=str(data["category"]),
                        image_path=self.pie_root / "annotation_images" / str(data["rel_path"]),
                        src_prompt=str(data["src_prompt"]),
                        tar_prompt=str(data["tar_prompt"]),
                        mask512=data["mask512"].astype(bool),
                        mask64=data["mask64"].astype(np.float32),
                        dist64=data["dist64"].astype(np.float32),
                    )
        if entry is None:
            rel_path = str(base.image_path.relative_to(self.pie_root / "annotation_images"))
            mask512 = mask_decode(base.rle_mask).astype(bool)
            mask64 = build_mask64(mask512)
            dist64 = build_dist64(mask64)
            entry = PIESample(
                image_id=image_id,
                category=base.editing_type_id,
                image_path=base.image_path,
                src_prompt=base.original_prompt,
                tar_prompt=base.editing_prompt,
                mask512=mask512,
                mask64=mask64,
                dist64=dist64,
            )
            if self.cache_dir is not None:
                path = self.cache_dir / f"{image_id}.npz"
                np.savez_compressed(
                    path,
                    category=np.array(entry.category),
                    rel_path=np.array(rel_path),
                    src_prompt=np.array(entry.src_prompt),
                    tar_prompt=np.array(entry.tar_prompt),
                    mask512=mask512,
                    mask64=mask64,
                    dist64=dist64,
                )
        self._cache[image_id] = entry
        return entry

    def category_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for image_id in self.image_ids:
            category = self._mapping[image_id].editing_type_id
            counts[category] = counts.get(category, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: int(kv[0])))
