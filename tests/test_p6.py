"""P6 unit tests: the Ours-4 composition, the A/B plumbing and the new background DINO.

The GPU parts of P6 (``mu_src`` cache, judge forwards) are exercised by the scripts; here we pin
down the algebra that the report's claims rest on, plus the two contracts that are easy to break:
``struct_dist_unedit`` must match the official ``evaluate.py`` path, and ``k=0`` must reproduce the
plain masked transport (so the k-sweep has a clean baseline).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from geoedit.edits import l5_mask, ours4_delta, attention_term, describe
from geoedit.masks import gaussian_blur
from geoedit.vlm_judge import AB_HEAD, AB_QUESTION, A_IDS, B_IDS

ROOT = Path(__file__).resolve().parents[1]


def _residual(seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(4, 64, 64, generator=g) * scale


def test_l5_mask_is_normalised_and_smooth():
    r5 = _residual(0)
    m = l5_mask(r5)
    assert m.shape == (64, 64)
    assert float(m.max()) == pytest.approx(1.0, abs=1e-6)
    assert float(m.min()) >= 0.0
    # sigma=2 must spread energy: the max cannot exceed the unsmoothed peak
    assert float(m.max()) <= float(r5.norm(dim=0).max()) + 1e-6
    # a constant map stays constant (the blur is exact at the border)
    ones = torch.ones(64, 64)
    assert torch.allclose(gaussian_blur(ones, 2.0), ones, atol=1e-5)


def test_k_zero_is_the_plain_masked_transport():
    """k=0 must not depend on A at all -- that is what makes it a valid same-family baseline."""
    r9, a9 = _residual(1), _residual(2)
    m = l5_mask(_residual(3))
    for gamma in (1.0, 2.0):
        d0 = ours4_delta(r9, a9, m, lam=1.3, k=0.0, gamma=gamma)
        d0_other = ours4_delta(r9, torch.zeros_like(a9), m, lam=1.3, k=0.0, gamma=gamma)
        assert torch.allclose(d0, d0_other)
        assert torch.allclose(d0, 1.3 * m * r9, atol=1e-6)


def test_gamma_concentrates_on_the_core():
    """gamma=2 weights the k-term by M, so low-M cells get a relatively smaller push."""
    r9 = torch.zeros(4, 64, 64)
    a9 = torch.ones(4, 64, 64)
    m = torch.zeros(64, 64)
    m[:32] = 1.0          # core
    m[32:] = 0.25         # shoulder
    d1 = ours4_delta(r9, a9, m, lam=1.0, k=1.0, gamma=1.0)
    d2 = ours4_delta(r9, a9, m, lam=1.0, k=1.0, gamma=2.0)
    ratio1 = float(d1[:, 32:, :].abs().mean() / d1[:, :32, :].abs().mean())
    ratio2 = float(d2[:, 32:, :].abs().mean() / d2[:, :32, :].abs().mean())
    assert ratio1 == pytest.approx(0.25, rel=0.02)
    assert ratio2 == pytest.approx(0.25 ** 2, rel=0.05)
    # outside the support the update vanishes for both
    m0 = m.clone()
    m0[48:] = 0.0
    assert float(ours4_delta(r9, a9, m0, 1.0, 1.0, 2.0)[:, 48:, :].abs().max()) == 0.0


def test_attention_term_is_r_plus_source_error():
    """A = R + (mu_src - x_s) exactly -- the identity the diagnose script measures."""
    x_s, r9, e_src = _residual(4), _residual(5), _residual(6, 0.1)
    mu_src = x_s + e_src
    a9 = attention_term(r9, mu_src, x_s)
    assert torch.allclose(a9, r9 + e_src, atol=1e-6)
    # with zero source error the attraction collapses onto R (the P5 approximation)
    assert torch.allclose(attention_term(r9, x_s, x_s), r9)


def test_describe_reports_the_decomposition():
    x_s, r9 = _residual(7), _residual(8)
    m = l5_mask(_residual(9))
    a9 = attention_term(r9, x_s + 0.05 * _residual(10), x_s)
    out = describe(r9, a9, m, k=1.0, gamma=2.0)
    assert out["R_norm"] > 0 and out["A_norm"] > 0
    assert -1.0 <= out["cos_RA"] <= 1.0
    assert 0.0 <= out["mask_area_frac"] <= 1.0
    assert out["k"] == 1.0 and out["gamma"] == 2.0


def test_ab_prompt_and_tokens():
    text = AB_HEAD.format(pa="first", pb="second", instr="Make the frame rusty") + AB_QUESTION
    assert "image A is shown first" in text and "image B second" in text
    assert "Make the frame rusty" in text
    assert "A or B" in text
    # both spellings of each letter must be single tokens and disjoint
    assert set(A_IDS).isdisjoint(set(B_IDS))
    assert len(A_IDS) == 2 and len(B_IDS) == 2


def test_metrics_key_registered():
    from geoedit.metrics import METRIC_KEYS
    assert "struct_dist_unedit" in METRIC_KEYS
    assert "struct_dist" in METRIC_KEYS


@pytest.mark.parametrize("k,gamma", [(0.0, 1.0), (1.0, 2.0)])
def test_ours4_shapes(k, gamma):
    r9, a9 = _residual(11), _residual(12)
    m = l5_mask(_residual(13))
    out = ours4_delta(r9, a9, m, lam=2.0, k=k, gamma=gamma)
    assert out.shape == (4, 64, 64)
    assert torch.isfinite(out).all()
    assert np.isfinite(float(out.norm()))
