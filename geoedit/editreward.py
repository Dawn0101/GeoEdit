"""Thin wrapper around the local EditReward checkpoint (Qwen2.5-VL-7B + two reward heads).

Reimplements only what inference needs, following the official code
(``TIGER-AI-Lab/EditReward``, ``EditReward/inference_vl_edit.py`` and
``EditReward/model/qwen2_5_vl_trainer.py``):

* two heads, ``head_0`` scores "Instruction Following & Semantic Fidelity" (1-4),
  ``head_1`` scores "Visual Quality & Realism" (1-4);
* the prompt is ``INSTRUCTION_EDIT_*`` + ``"Please provide the overall ratings of this image: <|Reward|>"``;
* the head is applied to **every** token's hidden state and pooled at the ``<|Reward|>`` position;
* ``min_pixels = max_pixels = 256*28*28``;
* each head outputs ``(mean, log_sigma)``; the released score is ``mean`` (it is a
  Bradley-Terry reward, so only **differences** are calibrated).

**P5-§1.1 bug fix.**  The P4 version set ``config.vocab_size = 151666`` on the *outer*
multimodal config only.  ``Qwen2_5_VLConfig.vocab_size`` is a convenience field; the text model
reads ``config.text_config.vocab_size``, which stayed at 152064.  The checkpoint's
``embed_tokens``/``lm_head`` are ``[151666, 3584]``, so ``from_pretrained`` reported a size
mismatch and -- because ``ignore_mismatched_sizes=True`` -- **reinitialised both at random**.
The vision tower (shape-identical) still loaded, so the model saw images but read every text
token through random embeddings; the scores in ``results/p4/e1d`` came from such a model and are
therefore void.  The override now goes into ``text_config`` as well, loading is strict, and the
embedding size is asserted against the tokenizer length.

The paper: EditReward, arXiv 2509.26346 (ICLR 2026).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

DEFAULT_PATH = Path("/data/lx/Model/Qwen2.5-VL-7B-SFT")
REWARD_TOKEN = "<|Reward|>"
MIN_PIXELS = MAX_PIXELS = 256 * 28 * 28
# 151665 base tokenizer rows + ``<|Reward|>``; the checkpoint's embed_tokens/lm_head are this wide
CHECKPOINT_VOCAB = 151666

INSTRUCTION_EDIT_FOLLOWING = """
You are tasked with evaluating an edited image **in comparison with the original source image** based on **Instruction Following & Semantic Fidelity**, and assigning a score from 1 to 4, with 1 being the worst and 4 being the best.
This dimension focuses on how accurately, completely, and exclusively the model executed the given text instruction.

**Inputs Provided:**
- Source Image (before editing)
- Edited Image (after applying the instruction)
- Text Instruction

**Sub-Dimensions to Evaluate:**

- **Semantic Accuracy:**
  Assess whether the edited content accurately captures the semantics of the instruction. The edited result should precisely match the intended meaning. For example, if the instruction is "replace apples with oranges," the object must clearly be oranges, not other fruits.

- **Completeness of Editing:**
  Check whether **all parts** of the instruction are fully executed. For multi-step edits (e.g., "replace a red car with a blue bicycle"), both the color change and the object replacement must be done without omissions.

- **Exclusivity of Edit (No Over-Editing):**
  Ensure that only the requested parts are changed. The rest of the image (as seen in the source) should remain unaltered. For example, if the instruction only involves replacing an object, the background, lighting, and unrelated objects should not be unnecessarily modified.

**Scoring Criteria:**
- **4 (Very Good):** Perfectly accurate, complete, and exclusive execution of the instruction.
- **3 (Relatively Good):** Largely correct, but with minor omissions or slight over-editing.
- **2 (Relatively Poor):** Major misinterpretation, incomplete edits, or noticeable unintended changes.
- **1 (Very Poor):** Instruction ignored or completely wrong execution.

Text instruction - {text_prompt}
"""

INSTRUCTION_EDIT_QUALITY = """
You are tasked with evaluating an edited image **in comparison with the original source image** based on **Visual Quality & Realism**, and assigning a score from 1 to 4, with 1 being the worst and 4 being the best.
This dimension focuses on how realistic, artifact-free, and aesthetically appealing the edited image is, while remaining consistent with the source image.

**Inputs Provided:**
- Source Image (before editing)
- Edited Image (after applying the instruction)
- Text Instruction

**Sub-Dimensions to Evaluate:**

- **Plausibility & Physical Consistency:**
  Check whether the edit aligns with the laws of physics and the scene context. Lighting, shadows, reflections, perspective, size, and interactions with the environment should all appear natural compared to the source image.

- **Artifact-Free Quality:**
  Look for technical flaws such as blur, distortions, pixel misalignment, unnatural textures, or seams around edited regions. High-quality results should be free from such visible artifacts.

- **Aesthetic Quality:**
  Evaluate the overall harmony and visual appeal. The image should look natural, balanced, and pleasant. Colors, composition, and atmosphere should enhance the image rather than degrade it.

**Scoring Criteria:**
- **4 (Very Good):** Perfectly realistic, artifact-free, seamless, and aesthetically pleasing.
- **3 (Relatively Good):** Mostly realistic and clean, with only minor flaws that do not significantly distract.
- **2 (Relatively Poor):** Noticeable physical inconsistencies or visible artifacts that make the edit unnatural.
- **1 (Very Poor):** Severe artifacts, incoherent composition, or visually unusable result.

Text instruction - {text_prompt}
"""

SUFFIX = "\nPlease provide the overall ratings of this image: " + REWARD_TOKEN + "\n\nEND\n"


def _build_head(hidden: int = 3584) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden, 1024), nn.ReLU(), nn.Dropout(0.05),
        nn.Linear(1024, 16), nn.ReLU(), nn.Linear(16, 2),
    )


class EditRewardJudge:
    """Holds the 7B backbone + the two reward heads; ``score_batch`` returns (dim1, dim2)."""

    def __init__(self, path=DEFAULT_PATH, device: str = "cuda:0", dtype=torch.bfloat16):
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.dtype = dtype
        # The override MUST go into ``text_config`` (see the module docstring): the outer
        # ``config.vocab_size`` is not what the text model reads, and getting this wrong
        # silently random-initialises the token embeddings.
        config = AutoConfig.from_pretrained(str(path))
        config.vocab_size = CHECKPOINT_VOCAB
        config.text_config.vocab_size = CHECKPOINT_VOCAB
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(path), config=config, torch_dtype=dtype,
            ignore_mismatched_sizes=False).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(str(path), padding_side="right",
                                                       min_pixels=MIN_PIXELS,
                                                       max_pixels=MAX_PIXELS)
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [REWARD_TOKEN]})
        self.reward_token_id = self.processor.tokenizer.convert_tokens_to_ids(REWARD_TOKEN)
        n_embed = int(self.model.get_input_embeddings().weight.shape[0])
        if n_embed != len(self.processor.tokenizer) or n_embed != CHECKPOINT_VOCAB:
            raise RuntimeError(
                f"token embedding table ({n_embed}) does not match the tokenizer "
                f"({len(self.processor.tokenizer)}); the checkpoint would be partially random")

        state = load_file(str(Path(path) / "model.safetensors"), device="cpu")
        self.heads = nn.ModuleList([_build_head() for _ in range(2)])
        for i, head in enumerate(self.heads):
            for sub in ("0.weight", "0.bias", "3.weight", "3.bias", "5.weight", "5.bias"):
                key = f"rm_heads.head_{i}.{sub}"
                module_name, param = sub.split(".")
                layer = head[int(module_name)]
                getattr(layer, "weight" if param == "weight" else "bias").data.copy_(
                    state[key].float())
        self.heads = self.heads.to(device).eval()
        self.batch_size = 4

    def _build(self, items: Sequence[Tuple[object, object, str]], template: str):
        """Chat-template + processor batch; ``min_pixels == max_pixels`` keeps every image equal."""
        message_list = []
        for src, edited, text in items:
            message_list.append([{"role": "user", "content": [
                {"type": "image", "image": src, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
                {"type": "image", "image": edited, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": template.format(text_prompt=text) + SUFFIX},
            ]}])
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in message_list]
        images = []
        for src, edited, _text in items:
            images.extend([src, edited])
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    @torch.no_grad()
    def score_batch(self, items: Sequence[Tuple[object, object, str]],
                    dims: Sequence[int] = (0, 1)) -> List[Tuple[float, float]]:
        """``items`` = ``(source PIL, edited PIL, instruction)`` -> per item ``(dim1, dim2)``."""
        templates = {0: INSTRUCTION_EDIT_FOLLOWING, 1: INSTRUCTION_EDIT_QUALITY}
        scores = {i: [0.0] * len(items) for i in dims}
        for dim in dims:
            for start in range(0, len(items), self.batch_size):
                chunk = items[start:start + self.batch_size]
                batch = self._build(chunk, templates[dim])
                hidden = self.model.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"],
                    use_cache=False)[0]
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    logits = self.heads[dim](hidden.float())
                pooled = self._pool(logits, batch["input_ids"])
                for j, value in enumerate(pooled[:, 0].float().cpu().tolist()):
                    scores[dim][start + j] = float(value)
        return [(scores[dims[0]][i], scores[dims[-1]][i] if len(dims) > 1 else float("nan"))
                for i in range(len(items))]

    def _pool(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Mean of the logits at the ``<|Reward|>`` position (``reward_token="special"``)."""
        mask = input_ids == self.reward_token_id
        counts = mask.sum(dim=1)
        if not bool((counts == 1).all()):
            raise RuntimeError(f"expected exactly one reward token per sample, got {counts.tolist()}")
        return logits[mask]

    @torch.no_grad()
    def score_pairs(self, items, batch_size: int = 4) -> List[Tuple[float, float]]:
        self.batch_size = int(batch_size)
        return self.score_batch(items)
