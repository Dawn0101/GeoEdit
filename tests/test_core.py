"""CPU tests that do not need SD-Turbo weights (T-M1, T-M2, T-E1..T-E4, T-H1, T-H2, T-D*, T-S*)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage

from geoedit import edit_ops, halo, stats
from geoedit.data import PIEDataset, build_dist64, build_mask64
from geoedit.edit_ops import make_eps, picard, residual, source_mu, lam_edit, image_seed


# --------------------------------------------------------------------------- #
# A fake Gaussian denoiser: mu_c(z) = mu_c + W (z/a - mu_c)   (doc §1 / T-E4)
# --------------------------------------------------------------------------- #
class FakeGaussianModel:
    def __init__(self, mu_src, mu_tar, w_diag, t=0.9, num_timesteps=1000):
        self.mu = {"src": mu_src, "tar": mu_tar}
        self.w = w_diag
        self.num_train_timesteps = num_timesteps
        self.max_timestep = num_timesteps - 1
        self.prediction_type = "epsilon"
        self.nfe = 0
        abar = 0.0125
        self.abar = torch.full((num_timesteps,), abar)
        self.alphas_cumprod = self.abar

    def timestep_index(self, t):
        return int(round(t * self.max_timestep))

    def ab(self, t):
        a = float(self.abar[self.timestep_index(t)].sqrt())
        b = float((1 - self.abar[self.timestep_index(t)]).sqrt())
        return a, b

    def x0_pred(self, z, t, emb):
        self.nfe += 1
        a, b = self.ab(t if not torch.is_tensor(t) else float(t[0]))
        which = "src" if float(emb.reshape(-1)[0]) == 0.0 else "tar"
        mu = self.mu[which]
        return mu + self.w * (z / a - mu)


def _fake_case(dim=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mu_src = torch.randn(dim, 1, 1, 1, generator=gen)
    mu_tar = mu_src + 0.5 * torch.randn(dim, 1, 1, 1, generator=gen)
    w = torch.linspace(0.2, 0.9, dim).view(dim, 1, 1, 1)
    model = FakeGaussianModel(mu_src, mu_tar, w)
    x_s = torch.randn(dim, 1, 1, 1, generator=gen)
    eps = torch.randn(dim, 1, 1, 1, generator=gen)
    emb_src = torch.zeros(1, 1, 1)
    emb_tar = torch.ones(1, 1, 1)
    return model, x_s, eps, emb_src, emb_tar, mu_src, mu_tar, w


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
# T-E1..T-E4
# --------------------------------------------------------------------------- #
def test_t_e1_eps_is_shared_and_deterministic():
    e1 = make_eps("000000000001", 0, torch.device("cpu"), torch.float32)
    e2 = make_eps("000000000001", 0, torch.device("cpu"), torch.float32)
    e3 = make_eps("000000000001", 1, torch.device("cpu"), torch.float32)
    assert torch.equal(e1, e2)
    assert not torch.equal(e1, e3)
    assert image_seed("a", 0) != image_seed("b", 0)


def test_t_e2_picard_k1_equals_lam1():
    model, x_s, eps, emb_src, emb_tar, *_ = _fake_case()
    res = residual(model, x_s, 0.9, emb_src, emb_tar, eps)
    one_step = lam_edit(x_s, res, 1.0)
    result = picard(model, x_s, 0.9, emb_src, emb_tar, eps, k_max=1)
    assert torch.allclose(one_step, x_s + result.delta(1), atol=1e-6)


def test_t_e3_picard_nfe_is_one_plus_k():
    model, x_s, eps, emb_src, emb_tar, *_ = _fake_case()
    for k in (1, 3, 6):
        model.nfe = 0
        result = picard(model, x_s, 0.9, emb_src, emb_tar, eps, k_max=k)
        assert model.nfe == 1 + k
        assert result.nfe == 1 + k


def test_t_e4_picard_error_equals_w_power_k():
    model, x_s, eps, emb_src, emb_tar, mu_src, mu_tar, w = _fake_case(dim=8)
    delta_mu = mu_tar - mu_src
    for k in (1, 2, 3, 5):
        result = picard(model, x_s, 0.9, emb_src, emb_tar, eps, k_max=k)
        error = delta_mu - result.delta(k)
        expected = (w ** k) * delta_mu
        assert torch.allclose(error, expected, rtol=1e-4, atol=1e-6), f"K={k}: {error.flatten()[:3]} vs {expected.flatten()[:3]}"


# --------------------------------------------------------------------------- #
# T-H1 / T-H2
# --------------------------------------------------------------------------- #
def _mask_and_dist(size=64, lo=20, hi=40):
    mask = np.zeros((size, size), np.float32)
    mask[lo:hi, lo + 4:hi + 4] = 1.0
    return mask, build_dist64(mask)


def test_t_h1_synthetic_halo_is_detected():
    mask, dist = _mask_and_dist()
    bump = torch.from_numpy(mask)[None, None] * 0.1
    res = bump - 0.1 * halo.gaussian_blur(bump, 3.0)
    value, n_pix = halo.rho(res, mask, dist, band=(2, 6), sigma=3.0)
    assert n_pix > 50
    assert value < -0.5, f"rho={value}"


def test_t_h2_white_noise_has_no_halo():
    mask, dist = _mask_and_dist()
    rng = np.random.default_rng(7)
    values = []
    for _ in range(100):
        res = torch.from_numpy(rng.standard_normal((1, 4, 64, 64)).astype(np.float32))
        value, _ = halo.rho(res, mask, dist, band=(2, 6), sigma=3.0)
        values.append(value)
    assert abs(float(np.nanmean(values))) < 0.05


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
