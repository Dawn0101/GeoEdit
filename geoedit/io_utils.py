"""Run ids, latent I/O, resumable CSV append, shard merging, manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def git_hash(repo: Optional[Path] = None) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo) if repo else None,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def run_id(**fields) -> str:
    """Stable id from the fields that define a configuration, e.g. ``t0.9_s0_lam_1.0_raw``."""
    parts = []
    for key in sorted(fields):
        value = fields[key]
        if isinstance(value, float):
            value = f"{value:g}"
        parts.append(f"{key}{value}")
    return "_".join(parts)


def save_latent(path: Path, tensor: torch.Tensor) -> None:
    """Save a latent as fp16 ``.pt`` atomically (a killed run never leaves half a file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor.detach().to(torch.float16).cpu(), tmp)
    os.replace(tmp, path)


def load_latent(path: Path) -> torch.Tensor:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def append_rows(path: Path, rows: Sequence[Dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    """Append CSV rows, writing the header only when the file is new."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames) if fieldnames else list(rows[0].keys())
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def merge_shards(shard_paths: Iterable[Path], out_path: Path) -> int:
    """Concatenate shard CSVs into ``out_path`` (header from the first non-empty shard)."""
    merged: List[Dict[str, str]] = []
    for path in shard_paths:
        merged.extend(read_rows(path))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not merged:
        out_path.write_text("")
        return 0
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)
    return len(merged)


def write_manifest(path: Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "created": now(),
        "host": socket.gethostname(),
        "git": git_hash(),
        **payload,
    }
    path.write_text(json.dumps(body, indent=2, default=str) + "\n")


def write_json(path: Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def load_yaml(path: Path) -> Dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def env_snapshot() -> Dict[str, str]:
    import geoedit
    import scipy
    import torch

    try:
        import diffusers

        diffusers_version = diffusers.__version__
    except Exception:
        diffusers_version = "unknown"
    return {
        "torch": torch.__version__,
        "diffusers": diffusers_version,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda or "unknown",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "geoedit": getattr(geoedit, "__version__", "0.2.0"),
    }
