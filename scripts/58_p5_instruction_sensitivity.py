"""P5-§1.3 diagnostic: on *our* images, does the instruction text matter at all?

§1.3's ``E1a`` control showed that a residual built from **another image's prompt** still reaches
85% of V0's gain under the Qwen judge.  That is either

(a) "the edit doesn't follow any instruction, so the judge answers on image quality alone", or
(b) "the judge cannot read the instruction for these images, whatever the residual".

This probe separates them without touching the residual: keep the **image pair fixed** and only
change the **instruction text**.  If ``score(true) - score(other)`` is ~0, the instruction is being
ignored for these images (b); if it is clearly positive, the instruction is being read and the E1a
failure is about the residual being instruction-agnostic (a) -- which is a statement about the
method at ``lam=1``, not about the judge.

Variants tested: ``V0`` and ``E1b`` (same-envelope noise) at ``lam=1``, plus the unedited ``recon``
as a floor.  The "other" instruction is the P4 permutation partner (another category).

Outputs ``results/p5/diagnostics/p5_instruction_sensitivity.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import edit_ops  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.editreward import EditRewardJudge  # noqa: E402
from geoedit.io_utils import load_yaml, now, write_json  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402
from geoedit.vlm_judge import QwenYesNoJudge  # noqa: E402


def build_perm(dataset, pool) -> Dict[str, str]:
    rng = np.random.default_rng(20261125)
    by_cat: Dict[str, List[str]] = {}
    for i in pool:
        by_cat.setdefault(dataset[i].category, []).append(dataset[i].image_id)
    perm = {}
    for i in pool:
        s = dataset[i]
        choices = [x for c, v in by_cat.items() if c != s.category for x in v]
        perm[s.image_id] = str(rng.choice(choices))
    return perm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n", type=int, default=40, help="images (spread over the 200-image set)")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:4")
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = [i for i in range(len(dataset)) if dataset[i].category not in ("8", "9")]
    eval_idx = [pool[i] for i in np.linspace(0, len(pool) - 1, 200).round().astype(int).tolist()]
    picked = eval_idx[:: max(1, len(eval_idx) // args.n)][: args.n]
    perm = build_perm(dataset, pool)

    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"], vae_only=True)
    er = EditRewardJudge(device=args.device)
    qwen = QwenYesNoJudge(device=args.device, backbone=er)
    a_root = ROOT / base["output_root"] / "p3" / "a"
    out_dir = ROOT / base["output_root"] / "p5" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for k, index in enumerate(picked):
        sample = dataset[index]
        other = dataset[perm[sample.image_id]]
        x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt",
                         weights_only=True).float().to(model.device)
        bundle = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
        R0 = bundle["t0.9_s0"].float().to(model.device)
        g = torch.Generator(device="cpu").manual_seed(edit_ops.image_seed(sample.image_id, 777))
        xi = torch.randn(R0.shape, generator=g, dtype=torch.float32).to(model.device)
        envelope = R0.abs() * xi
        noise = envelope / envelope.norm().clamp_min(1e-12) * float(R0.norm())
        latents = {"recon": x_s, "V0": x_s + args.lam * R0, "E1b": x_s + args.lam * noise}
        decoded = {name: Image.fromarray(model.decode(lat if lat.dim() == 4 else lat.unsqueeze(0))[0])
                   for name, lat in latents.items()}
        src = sample.image()

        for name, img in decoded.items():
            pair_t = [(src, img, sample.instruction)]
            pair_o = [(src, img, other.instruction)]
            q_t = qwen.score_following(pair_t, batch_size=1)[0]
            q_o = qwen.score_following(pair_o, batch_size=1)[0]
            er_t = er.score_pairs(pair_t, batch_size=1)[0][0]
            er_o = er.score_pairs(pair_o, batch_size=1)[0][0]
            rows.append({"image_id": sample.image_id, "category": sample.category,
                         "variant": name, "lam": args.lam,
                         "instruction": sample.instruction,
                         "other_instruction": other.instruction,
                         "q_true": q_t, "q_other": q_o, "q_delta": q_t - q_o,
                         "er_true": er_t, "er_other": er_o, "er_delta": er_t - er_o})
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(picked)}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "p5_instruction_sensitivity.csv", index=False)
    table = (frame.groupby("variant")[["q_true", "q_other", "q_delta", "er_true", "er_other",
                                       "er_delta"]].mean().round(3))
    print(table.to_string())
    write_json(out_dir / "p5_instruction_sensitivity.json", {
        "created": now(), "n_images": int(frame["image_id"].nunique()), "lam": args.lam,
        "means": table.to_dict(orient="index"),
        "reading": "q_delta = score with this image's own instruction minus the score with another "
                   "image's instruction, image pair held fixed. Near zero => the instruction text "
                   "does not change the judge's answer on these images.",
    })
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
