"""P5-§1.2: a second, independent VLM judge -- Qwen2.5-VL, two images, Yes/No log-probability.

EditReward is a Bradley-Terry reward head bolted onto Qwen2.5-VL; here the plain backbone is
used instead, with a question that has a literal answer, so the number has a fixed meaning:

    following = log P(Yes) - log P(No)  for "does the edited image follow the instruction?"

``P(Yes)`` is summed over every byte-level spelling of the answer (``Yes``/`` Yes``/``yes``/
`` YES``; likewise for ``No``) with ``logsumexp``, evaluated at the first answer position.  No
generation is performed -- one forward pass per item, and only the two groups of logits are read.

Both image orders are supported.  ``order="forward"`` puts the source image first (the order the
EditReward prompts use); ``order="swapped"`` puts the edited image first **and swaps the wording**
("first image"/"second image"), so the semantics of the question are identical and only the
position changes -- the agreement between the two orders is therefore a position-bias check.

The instruction is PIE-Bench's own ``editing_instruction`` (a sentence, e.g. "Make the frame of
the bike rusty"), not the target caption.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import torch

DEFAULT_PATH = Path("/data/lx/Model/Qwen2.5-VL-7B-SFT")
CHECKPOINT_VOCAB = 151666  # 151665 base rows + <|Reward|>; keeps the embedding table loadable

# every spelling of the answer that tokenises to a single token in the Qwen2.5 tokenizer
YES_IDS = (9454, 7414, 9693, 14080)  # "Yes", " Yes", "yes", " YES"
NO_IDS = (2753, 2308, 2152, 5664)  # "No", " No", "no", " NO"

_HEAD = ('The {first} image is the original photo. The {second} image is the result of applying '
         'an edit to it.\nThe editing instruction was: "{instr}"\n')
FOLLOW_QUESTION = "Question: does the {topic} image follow the instruction? Answer with one word, Yes or No."
KEEP_QUESTION = ('Question: apart from "{obj}", is the rest of the {topic} image identical to the '
                 'other image? Answer with one word, Yes or No.')


class QwenYesNoJudge:
    """``score_following`` / ``score_keep`` -> logP(Yes) - logP(No) per item."""

    def __init__(self, path: Union[str, Path, None] = DEFAULT_PATH, device: str = "cuda:0",
                 dtype: torch.dtype = torch.bfloat16, backbone=None) -> None:
        """``backbone``: an ``EditRewardJudge`` -- reuse its 7B weights and processor.

        Both judges are the same Qwen2.5-VL checkpoint (EditReward only adds two heads), so
        sharing one copy of the backbone halves the GPU footprint.
        """
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        self.device = device
        if backbone is not None:
            self.model = backbone.model
            self.processor = backbone.processor
            return
        config = AutoConfig.from_pretrained(str(path))
        config.vocab_size = CHECKPOINT_VOCAB
        config.text_config.vocab_size = CHECKPOINT_VOCAB
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(path), config=config, torch_dtype=dtype, ignore_mismatched_sizes=False,
            attn_implementation="sdpa").to(device).eval()
        self.processor = AutoProcessor.from_pretrained(str(path), padding_side="right")
        n_embed = int(self.model.get_input_embeddings().weight.shape[0])
        # the checkpoint carries one extra row for <|Reward|>, which this judge never uses
        if n_embed not in (len(self.processor.tokenizer), len(self.processor.tokenizer) + 1):
            raise RuntimeError("token embeddings and tokenizer disagree -- the checkpoint would be "
                               "partially random")

    def _prompt(self, first: str, second: str, topic: str, question: str, instr: str,
                obj: str) -> str:
        return (_HEAD.format(first=first, second=second, instr=instr)
                + question.format(topic=topic, obj=obj))

    def _build(self, texts: Sequence[str], images: Sequence[Tuple[object, object]]):
        messages = [[{"role": "user", "content": [
            {"type": "image", "image": a}, {"type": "image", "image": b},
            {"type": "text", "text": t}]}] for t, (a, b) in zip(texts, images)]
        flat = []
        for item in messages:
            flat.extend(c["image"] for c in item[0]["content"] if c["type"] == "image")
        batch = self.processor(
            text=[self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                  for m in messages],
            images=flat, padding=True, return_tensors="pt")
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    @torch.no_grad()
    def _score(self, texts: Sequence[str], images: Sequence[Tuple[object, object]],
               batch_size: int = 4) -> List[float]:
        out: List[float] = []
        for start in range(0, len(texts), batch_size):
            batch = self._build(texts[start:start + batch_size], images[start:start + batch_size])
            logits = self.model(**batch, use_cache=False).logits.float()
            # right padding -> the first answer position is the last real token
            lengths = (batch["attention_mask"].sum(dim=1) - 1).tolist()
            rows = logits[torch.arange(logits.shape[0]), torch.tensor(lengths, device=logits.device)]
            logp = torch.log_softmax(rows, dim=-1)
            yes = torch.logsumexp(logp[:, list(YES_IDS)], dim=-1)
            no = torch.logsumexp(logp[:, list(NO_IDS)], dim=-1)
            out.extend((yes - no).cpu().tolist())
        return out

    def score_following(self, items, batch_size: int = 4, order: str = "forward") -> List[float]:
        """``items`` = ``(source PIL, edited PIL, instruction)`` -> logP(Yes) - logP(No)."""
        return self.score_pairs(items, "following", batch_size=batch_size, order=order)

    def score_keep(self, items, batch_size: int = 4, order: str = "forward",
                   objects: Sequence[str] = ()) -> List[float]:
        return self.score_pairs(items, "keep", batch_size=batch_size, order=order, objects=objects)

    def score_pairs(self, items, kind: str = "following", batch_size: int = 4,
                    order: str = "forward", objects: Sequence[str] = ()) -> List[float]:
        if order not in ("forward", "swapped"):
            raise ValueError(order)
        texts, images = [], []
        for i, (src, edited, instr) in enumerate(items):
            obj = objects[i] if i < len(objects) else ""
            if order == "forward":
                first, second, topic, pair = "first", "second", "second", (src, edited)
            else:  # edited image first; the wording follows the position, not the role
                first, second, topic, pair = "second", "first", "first", (edited, src)
            question = FOLLOW_QUESTION if kind == "following" else KEEP_QUESTION
            texts.append(self._prompt(first, second, topic, question, instr, obj))
            images.append(pair)
        return self._score(texts, images, batch_size=batch_size)
