"""P10: in-layer spatial guidance for the ChordEdit UNet --- SCAD and MGA.

Both act on the **cross-attention** (``attn2``) of the *target* branch, before the velocity
prediction is formed, so they can change *where and how strongly* the text semantics enter instead
of merely trimming the output afterwards.

SCAD (ours, token-free)
    For the same image query ``Q``, the difference ``O(c_tar) - O(c_src)`` shares ``Q`` exactly, so
    it is pure text-semantics injection with zero cross-branch noise.  With a per-pixel mask ``m``:

    ``A``:  ``O~ = O_tar + alpha * m * (O_tar - O_src)``
    ``B``:  ``O~ = O_src + (1 + alpha) * m * (O_tar - O_src)``   (outside the mask the branch reads
                                                                    the *source* text features)

MGA (literature baseline)
    A de-meaned spatial bias added to the attention logits of the *target edit tokens* only,
    before the softmax:  ``logits[:, :, E_tar] += lam * (m - mean(m))``.

The module patches every ``attn2`` in the UNet with one processor that can do either (or both) and
falls back to the stock ``AttnProcessor2_0`` when inactive, so an inactive patch is bit-identical to
the unmodified pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

SPECIAL_IDS = {49406, 49407}  # CLIP <bos> / <eos>+pad


# --------------------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------------------- #
def plateau_mask(energy: torch.Tensor, lo: float = 0.40, hi: float = 0.65) -> torch.Tensor:
    """Quantile plateau mask: the top ``1-hi`` fraction saturates at 1, background at 0.

    ``(E - q_lo) / (q_hi - q_lo)`` clipped to [0,1] means every pixel above ``q_hi`` is exactly 1.0
    -- no interior attenuation like the max-normalised mask (whose mean inside the object was 0.61,
    and which got squared by the earlier composite).
    """
    flat = energy.flatten()
    ql = torch.quantile(flat, lo)
    qh = torch.quantile(flat, hi)
    return ((energy - ql) / (qh - ql + 1e-8)).clamp(0.0, 1.0)


def edit_token_indices(tokenizer, src_prompt: str, tar_prompt: str) -> List[int]:
    """Positions in the target prompt that are not shared with the source prompt (special tokens
    stripped).  Empty for pure deletions, in which case MGA simply does nothing."""
    ids_src = tokenizer(src_prompt, truncation=True, max_length=77).input_ids
    ids_tar = tokenizer(tar_prompt, truncation=True, max_length=77).input_ids
    keep = lambda ids: [t for t in ids if t not in SPECIAL_IDS]
    src_set, tar_keep = set(keep(ids_src)), keep(ids_tar)
    return [i for i, tok in enumerate(ids_tar) if tok not in SPECIAL_IDS and tok not in src_set]


# --------------------------------------------------------------------------------------- #
# attention processor
# --------------------------------------------------------------------------------------- #
class _State:
    """Everything the processor needs; one per UNet call, mutated between calls."""

    def __init__(self) -> None:
        self.active = False
        self.src_ctx: Optional[torch.Tensor] = None
        self.mask: Optional[torch.Tensor] = None      # [64,64] in [0,1]
        self.scad: Optional[str] = None               # "A" | "B" | None
        self.alpha: float = 0.0
        self.mga: float = 0.0
        self.edit_idx: Sequence[int] = ()
        # capture: attn2 token-columns and attn1 self-attention maps
        self.capture2: bool = False
        self.capture1: bool = False
        self.acc2: List[torch.Tensor] = []
        self.acc1: Optional[torch.Tensor] = None
        # per-layer cross-attention outputs, for the "relative difference" statistic r
        self.o_src: List[torch.Tensor] = []
        self.r_num: float = 0.0
        self.r_den: float = 0.0
        self.capture_o: Optional[str] = None
        self._o_seen: Optional[int] = None

    def clear_capture(self) -> None:
        self.acc2 = []
        self.acc1 = None
        self.o_src = []
        self.r_num = 0.0
        self.r_den = 0.0
        self._o_seen = None

    def reset(self) -> None:
        self.active = False
        self.scad = None
        self.alpha = 0.0
        self.mga = 0.0
        self.edit_idx = ()
        self.mask = None


def _heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    b, n, d = x.shape
    return x.view(b, n, heads, d // heads).permute(0, 2, 1, 3).reshape(b * heads, n, d // heads)


def _merge(x: torch.Tensor, heads: int, b: int) -> torch.Tensor:
    n = x.shape[1]
    d = x.shape[2]
    return x.view(b, heads, n, d).permute(0, 2, 1, 3).reshape(b, n, heads * d)


class SpatialAttnProcessor2_0:
    """Drop-in replacement for ``AttnProcessor2_0`` with optional SCAD / MGA modulation."""

    def __init__(self, state: _State) -> None:
        self.state = state

    # the stock path, used whenever the state is inactive (bit-identical to diffusers)
    @staticmethod
    def _stock(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        from diffusers.models.attention_processor import AttnProcessor2_0
        return AttnProcessor2_0()(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)

    @torch.no_grad()
    def _record_o(self, attn, hidden_states, encoder_hidden_states, st, store: bool) -> None:
        """Store O for the source call, or accumulate ||O_tar - O_src|| / ||O_tar|| for the target call."""
        hs = hidden_states
        if hs.ndim == 4:
            b, c, h, w = hs.shape
            hs = hs.view(b, c, h * w).transpose(1, 2)
        if attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        heads = attn.heads
        q = _heads(attn.to_q(hs), heads)
        k = _heads(attn.to_k(encoder_hidden_states), heads)
        v = _heads(attn.to_v(encoder_hidden_states), heads)
        logits = torch.bmm(q.float(), k.float().transpose(1, 2)) * float(attn.scale)
        o = torch.bmm(torch.softmax(logits, dim=-1).to(q.dtype), v)
        if store:
            st.o_src.append(o.detach())
            return
        if st._o_seen is None:
            st._o_seen = 0
        i = st._o_seen
        st._o_seen += 1
        if i < len(st.o_src):
            d = o - st.o_src[i].to(o.device)
            st.r_num += float(d.float().norm() ** 2)
            st.r_den += float(o.float().norm() ** 2)

    @torch.no_grad()
    def _record_cross(self, attn, hidden_states, encoder_hidden_states, st) -> None:
        """Record the head-averaged attention mass on the edit-token columns at 64x64."""
        hs = hidden_states
        if hs.ndim == 4:
            b, c, h, w = hs.shape
            hs = hs.view(b, c, h * w).transpose(1, 2)
        if attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        b, hw, _ = hs.shape
        heads = attn.heads
        q = _heads(attn.to_q(hs), heads)
        k = _heads(attn.to_k(encoder_hidden_states), heads)
        logits = torch.bmm(q.float(), k.float().transpose(1, 2)) * float(attn.scale)
        probs = torch.softmax(logits, dim=-1).mean(dim=0)          # [HW, 77]
        g = int(round(hw ** 0.5))
        cols = probs[:, list(st.edit_idx)].mean(dim=1).reshape(1, 1, g, g)
        st.acc2.append(F.interpolate(cols.float(), size=(64, 64), mode="bilinear",
                                     align_corners=False)[0, 0].clone())

    def _self_attn_capture(self, attn, hidden_states, attention_mask=None, **kw):
        """Record the (head-averaged) self-attention map at the 32x32 level, pass through unchanged."""
        st = self.state
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        out = self._stock(attn, hidden_states, None, attention_mask, **kw)
        if hidden_states.shape[1] == 1024 and st.acc1 is None:
            heads = attn.heads
            q = _heads(attn.to_q(hidden_states), heads)
            k = _heads(attn.to_k(hidden_states), heads)
            logits = torch.bmm(q.float(), k.float().transpose(1, 2)) * float(attn.scale)
            probs = torch.softmax(logits, dim=-1).mean(dim=0)
            st.acc1 = probs.detach()
        return out

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 temb=None, **kw):
        st = self.state
        if not getattr(attn, "is_cross_attention", False):
            if st.capture1:
                return self._self_attn_capture(attn, hidden_states, attention_mask, **kw)
            return self._stock(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)
        if not st.active:
            if st.capture_o == "src":
                self._record_o(attn, hidden_states, encoder_hidden_states, st, store=True)
            elif st.capture_o == "tar":
                self._record_o(attn, hidden_states, encoder_hidden_states, st, store=False)
            out = self._stock(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)
            if st.capture2 and st.edit_idx:
                self._record_cross(attn, hidden_states, encoder_hidden_states, st)
            return out
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        if attention_mask is not None:
            return self._stock(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)
        if attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        b, hw, _ = hidden_states.shape
        heads = attn.heads
        q = _heads(attn.to_q(hidden_states), heads)
        dummy = torch.zeros(1, 1, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        del dummy
        k_t = _heads(attn.to_k(encoder_hidden_states), heads)
        v_t = _heads(attn.to_v(encoder_hidden_states), heads)
        logits = torch.bmm(q.float(), k_t.float().transpose(1, 2)) * float(attn.scale)

        mask = None
        if st.mask is not None:
            g = int(round(hw ** 0.5))
            m = F.interpolate(st.mask[None, None].float(), size=(g, g), mode="bilinear",
                              align_corners=False)[0, 0].reshape(1, hw, 1)
            mask = m.to(device=logits.device, dtype=torch.float32)

        if st.mga and st.edit_idx:
            bias = float(st.mga) * (mask - mask.mean())
            idx = torch.tensor(list(st.edit_idx), device=logits.device, dtype=torch.long)
            logits[:, :, idx] = logits[:, :, idx] + bias.to(logits.dtype)
        probs = torch.softmax(logits, dim=-1).to(q.dtype)
        out = torch.bmm(probs, v_t)

        if st.scad and st.src_ctx is not None:
            ctx = st.src_ctx
            if ctx.shape[0] != encoder_hidden_states.shape[0]:
                ctx = ctx.expand(encoder_hidden_states.shape[0], -1, -1)
            k_s = _heads(attn.to_k(ctx), heads)
            v_s = _heads(attn.to_v(ctx), heads)
            ls = torch.bmm(q.float(), k_s.float().transpose(1, 2)) * float(attn.scale)
            out_s = torch.bmm(torch.softmax(ls, dim=-1).to(q.dtype), v_s)
            d = out - out_s
            if st.scad == "A":
                out = out + float(st.alpha) * mask.to(out.dtype) * d
            else:
                out = out_s + (1.0 + float(st.alpha)) * mask.to(out.dtype) * d

        out = _merge(out, heads, b)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if input_ndim == 4:
            out = out.transpose(1, 2).reshape(b, -1, h, w)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def install(unet) -> _State:
    """Replace every ``attn2`` processor; returns the shared state."""
    state = _State()
    for module in unet.modules():
        if module.__class__.__name__ == "Attention":
            module.processor = SpatialAttnProcessor2_0(state)
    return state


def set_call(state: _State, *, src_ctx=None, mask=None, scad=None, alpha=0.0, mga=0.0,
             edit_idx: Sequence[int] = ()) -> None:
    state.src_ctx = src_ctx
    state.mask = mask
    state.scad = scad
    state.alpha = float(alpha)
    state.mga = float(mga)
    state.edit_idx = tuple(edit_idx)
    state.active = bool(mask is not None and (scad or mga))
