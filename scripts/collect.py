"""Collect run bundles into one markdown table, with the paper's numbers for reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from geoedit.cache import RESULTS

# ChordEdit paper, PIE-Bench, sd-turbo.  ``PSNR`` in that table is the PIE-Bench
# convention, i.e. computed on the background after masking (``psnr_unedit``).
PAPER = {
    "B0_chord_noprox": {"psnr_unedit": 23.89, "clip_edited": 21.87, "lpips_unedit": 0.0884},
    "B1_chord": {"psnr_unedit": 22.20, "clip_edited": 22.96, "lpips_unedit": 0.1283},
}

COLUMNS = ("psnr_unedit", "lpips_unedit", "clip_edited", "clip_whole", "struct_dist",
           "psnr", "lpips", "ssim")


def load(run_id: str) -> Optional[Dict]:
    path = RESULTS / run_id / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["B0_chord_noprox", "B1_chord"])
    ap.add_argument("--out", type=Path, default=Path("docs/summary_table.md"))
    args = ap.parse_args()

    header = "| run | n | " + " | ".join(COLUMNS) + " |"
    sep = "|" + "---|" * (len(COLUMNS) + 2)
    lines = [header, sep]
    for run_id in args.runs:
        summary = load(run_id)
        if summary is None:
            lines.append(f"| {run_id} | — | " + " | ".join("—" for _ in COLUMNS) + " |")
            continue
        overall = summary["overall"]
        cells = [f"{overall[c]:.3f}" if overall.get(c) is not None else "nan" for c in COLUMNS]
        lines.append(f"| {run_id} | {summary['n_images']} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Paper (ChordEdit, PIE-Bench, sd-turbo):")
    lines.append("")
    lines.append("| run | " + " | ".join(c for c in ("psnr_unedit", "lpips_unedit", "clip_edited")) + " |")
    lines.append("|" + "---|" * 4)
    for run_id, values in PAPER.items():
        cells = [f"{values[c]:.3f}" for c in ("psnr_unedit", "lpips_unedit", "clip_edited")]
        lines.append(f"| {run_id} | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)


if __name__ == "__main__":
    main()
