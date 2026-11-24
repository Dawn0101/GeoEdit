"""On-disk cache layout for ChordEdit proposals.

Every experiment after T2 reuses the same proposals, so they are computed once and
stored per image::

    cache/proposals/<image_id>.npz      x_src, y, y_prox, noise   (fp32, scaled latents)
    cache/images/B0_chord_noprox/<image_id>.jpg
    cache/images/B1_chord/<image_id>.jpg

Storing the *scaled* latents (not the decoded images alone) is what makes the OT sweep
cheap: a config only needs ``sigma`` plus one proximal UNet call, and the transport step
is never repeated.  ``noise`` is stored because both the transport and the prox use it;
sharing it across configs is what keeps the comparison fair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
PROPOSALS = CACHE / "proposals"
IMAGES = CACHE / "images"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

_LATENT_KEYS = ("x_src", "y", "y_prox", "noise")


def proposal_path(image_id: str) -> Path:
    return PROPOSALS / f"{image_id}.npz"


def image_dir(run_id: str) -> Path:
    return IMAGES / run_id


def image_path(run_id: str, image_id: str) -> Path:
    return image_dir(run_id) / f"{image_id}.jpg"


def save_proposal(image_id: str, arrays: Dict[str, np.ndarray], meta: Dict[str, float]) -> None:
    PROPOSALS.mkdir(parents=True, exist_ok=True)
    payload = {k: arrays[k].astype(np.float32) for k in _LATENT_KEYS}
    payload["meta"] = np.array([meta.get("transport_s", 0.0), meta.get("prox_s", 0.0), meta.get("seed", 0.0)])
    np.savez_compressed(proposal_path(image_id), **payload)


def load_proposal(image_id: str) -> Dict[str, np.ndarray]:
    with np.load(proposal_path(image_id)) as data:
        return {k: data[k] for k in _LATENT_KEYS}


def has_proposal(image_id: str) -> bool:
    path = proposal_path(image_id)
    return path.exists() and path.stat().st_size > 0
