"""CPU tests that do not need SD-Turbo weights (T-M1, T-M2, T-E1, T-D*, T-S1, T-S2).

v2's T-E2..T-E4 (Picard) and T-H1/T-H2 (halo) went with those modules to the ``v2`` branch;
P3's own tests are in ``test_p3.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from geoedit import stats
from geoedit.data import PIEDataset, build_dist64, build_mask64
from geoedit.edit_ops import image_seed, make_eps


# --------------------------------------------------------------------------- #
# T-M1 / T-M2
# --------------------------------------------------------------------------- #
def test_t_m1_a2_plus_b2_is_one():
    from diffusers import DDPMScheduler

    scheduler = DDPMScheduler.from_pretrained("/data/lx/Model/SD-Turbo/scheduler_ddpm")
    ac = scheduler.alphas_cumprod
    assert torch.allclose(ac + (1 - ac), torch.ones_like(ac), atol=1e-6)


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_t_m2_x0_conversion_roundtrip(prediction_type):
    gen = torch.Generator().manual_seed(3)
    x0 = torch.randn(2, 4, 8, 8, generator=gen)
    eps = torch.randn(2, 4, 8, 8, generator=gen)
    for abar in (0.9, 0.2, 0.0125):
        a, b = float(np.sqrt(abar)), float(np.sqrt(1 - abar))
        z = a * x0 + b * eps
        # epsilon: the network predicts eps; v_prediction: v = a*eps - b*x0
        network_out = eps if prediction_type == "epsilon" else a * eps - b * x0
        if prediction_type == "epsilon":
            recovered = (z - b * network_out) / a
        else:
            recovered = a * z - b * network_out
        assert torch.allclose(recovered, x0, atol=1e-5)


# --------------------------------------------------------------------------- #
# T-E1
# --------------------------------------------------------------------------- #
def test_t_e1_eps_is_shared_and_deterministic():
    e1 = make_eps("000000000001", 0, torch.device("cpu"), torch.float32)
    e2 = make_eps("000000000001", 0, torch.device("cpu"), torch.float32)
    e3 = make_eps("000000000001", 1, torch.device("cpu"), torch.float32)
    assert torch.equal(e1, e2)
    assert not torch.equal(e1, e3)
    assert image_seed("a", 0) != image_seed("b", 0)


# --------------------------------------------------------------------------- #
# T-D1 / T-D2
# --------------------------------------------------------------------------- #
def test_t_d1_dataset_has_700_samples_and_10_categories():
    dataset = PIEDataset(cache_dir=None)
    assert len(dataset) == 700
    counts = dataset.category_counts()
    assert len(counts) == 10
    assert sum(counts.values()) == 700


def test_t_d2_mask_decode_matches_official_and_pooling_is_consistent():
    from geoedit.pie import mask_decode

    dataset = PIEDataset(cache_dir=None)
    sample = dataset[0]
    decoded = mask_decode(dataset._mapping[sample.image_id].rle_mask)
    assert decoded.shape == (512, 512)
    assert decoded.dtype == np.uint8 or decoded.dtype == np.bool_
    assert decoded[0, :].all() and decoded[-1, :].all() and decoded[:, 0].all() and decoded[:, -1].all()
    mask64 = build_mask64(decoded.astype(bool))
    assert mask64.shape == (64, 64)
    assert set(np.unique(mask64)).issubset({0.0, 1.0})
    dist = build_dist64(mask64)
    if mask64.any():
        assert dist[mask64 > 0.5].max() == 0.0
        assert dist.max() > 0


# --------------------------------------------------------------------------- #
# T-S1 / T-S2
# --------------------------------------------------------------------------- #
def test_t_s1_bootstrap_coverage():
    rng = np.random.default_rng(0)
    covered = 0
    trials = 200
    for _ in range(trials):
        sample = rng.normal(5.0, 1.0, 100)
        ci = stats.boot_ci(sample, n_boot=400, rng=rng)
        covered += int(ci["lo"] <= 5.0 <= ci["hi"])
    coverage = covered / trials
    assert 0.90 <= coverage <= 0.99, coverage


def test_t_s2_interpolation_and_out_of_range():
    clip = [1.0, 2.0, 3.0]
    psnr = [20.0, 21.0, 22.0]
    assert abs(stats.interp_curve(clip, psnr, 1.5).psnr_curve_at - 20.5) < 1e-9
    assert stats.interp_curve(clip, psnr, 3.5).out_of_range
    assert stats.interp_curve(clip, psnr, 0.5).out_of_range
    delta = stats.delta_vs_curve(clip, psnr, 2.0, 25.0)
    assert abs(delta.delta - 4.0) < 1e-9 and not delta.out_of_range
