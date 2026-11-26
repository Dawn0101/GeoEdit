"""P5-§1.1: is our EditReward usage identical to the official one?

The official repo (TIGER-AI-Lab/EditReward) was cloned to /tmp/EditReward_off and read:

* ``EditReward/inference_vl_edit.py`` builds, per sample, a single user message with
  ``[src image, edited image, text]`` -- source **first**, edited **second** -- and appends
  ``prompt_with_special_token`` = ``"\nPlease provide the overall ratings of this image: <|Reward|>\n\nEND\n"``;
* ``ranknet_multi_head`` runs **two separate batches**: ``batch_dim1`` with
  ``INSTRUCTION_EDIT_FOLLOWING`` and ``batch_dim2`` with ``INSTRUCTION_EDIT_QUALITY``;
* the head is ``Linear(3584,1024)-ReLU-Dropout(.05)-Linear(1024,16)-ReLU-Linear(16,2)`` applied to
  every hidden state and pooled at ``<|Reward|>``; output = ``(mean, log_sigma)`` and the
  released score is ``mean``;
* ``min_pixels = max_pixels = 256*28*28``.

This probe checks the one thing that cannot be read off the source: whether the *loaded weights*
behave like the official model.  It scores the official demo images (``assets/examples``) for the
official demo instruction ("Add a green bowl on the branch"), including the degenerate pair
``(src, src)`` and the swapped order ``(target, src)`` to expose a position bias.

A correct reward model must give ``(src, target_k) > (src, src)`` for a target that does contain
the bowl.  If ``(src, src)`` wins, either the loading is wrong or the model itself prefers no edit.

Usage: ``python scripts/51_p5_judge_probe.py --device cuda:4``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.editreward import REWARD_TOKEN, EditRewardJudge  # noqa: E402
from geoedit.io_utils import now, write_json  # noqa: E402

EXAMPLE_DIR = Path("/tmp/EditReward_off/assets/examples")
INSTRUCTION = "Add a green bowl on the branch"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default="results/p5/judge_probe.json")
    args = parser.parse_args()

    judge = EditRewardJudge(path=args.model or EditRewardJudge.__init__.__defaults__[0],
                            device=args.device)
    tok = judge.processor.tokenizer
    print(f"tokenizer len={len(tok)}  <|Reward|> id={judge.reward_token_id}  "
          f"embed rows={judge.model.get_input_embeddings().weight.shape[0]}")

    src = Image.open(EXAMPLE_DIR / "source_img_1.png").convert("RGB")
    targets = {f"target_{i}": Image.open(EXAMPLE_DIR / f"target_img_{i}.png").convert("RGB")
               for i in (1, 2, 3)}
    print("sizes:", src.size, {k: v.size for k, v in targets.items()})

    pairs = [("src_src", src, src)]
    pairs += [(k, src, v) for k, v in targets.items()]
    pairs += [(f"{k}_swapped", v, src) for k, v in targets.items()]
    items = [(a, b, INSTRUCTION) for _n, a, b in pairs]
    scores = judge.score_pairs(items, batch_size=4)

    rows = [{"pair": n, "dim1": s[0], "dim2": s[1]} for (n, _a, _b), s in zip(pairs, scores)]
    for r in rows:
        print(f"  {r['pair']:18s} dim1(following)={r['dim1']:+.4f}  dim2(quality)={r['dim2']:+.4f}")

    src_src = rows[0]["dim1"]
    gains = {r["pair"]: r["dim1"] - src_src for r in rows[1:] if not r["pair"].endswith("_swapped")}
    print("\ndim1 gain over (src,src):",
          {k: round(v, 4) for k, v in gains.items()})
    verdict = "USAGE_OK" if any(v > 0 for v in gains.values()) else "MODEL_PREFERS_NO_EDIT"
    print(f"verdict: {verdict}")

    write_json(ROOT / args.out, {
        "created": now(), "instruction": INSTRUCTION, "device": args.device,
        "reward_token": REWARD_TOKEN, "reward_token_id": int(judge.reward_token_id),
        "tokenizer_len": len(tok), "embed_rows": int(judge.model.get_input_embeddings().weight.shape[0]),
        "rows": rows, "gains_over_src_src": gains, "verdict": verdict,
        "official_reference": "TIGER-AI-Lab/EditReward @ /tmp/EditReward_off (cloned 2026-11-26)",
    })
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
