"""P3 unit tests (doc §7): T-K1..T-K4, T-N1..T-N3, T-R1, T-S3, and T-X3 (GPU).

Everything except the ``gpu``-marked tests runs on CPU in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from geoedit import masks, noise_ops, reward, stats
from geoedit.edit_ops import residual_from_mu, source_mu


# --------------------------------------------------------------------------- #
# A fake Gaussian denoiser with a diagonal W:  mu_c(z) = mu_c + W (z/a - mu_c)
# --------------------------------------------------------------------------- #
class FakeDiagonalModel:
    """One fixed branch.  ``w`` and ``mu`` are ``[1,4,1,1]`` so they broadcast to the latent."""

    def __init__(self, mu, w, abar=0.0125, num_timesteps=1000):
        self.mu = mu
        self.w = w
        self.num_train_timesteps = num_timesteps
        self.max_timestep = num_timesteps - 1
        self.prediction_type = "epsilon"
        self.nfe = 0
        self.abar = torch.full((num_timesteps,), abar)

    def timestep_index(self, t):
        return int(round(float(t) * self.max_timestep))

    def ab(self, t):
        tau = self.timestep_index(t)
        return float(self.abar[tau].sqrt()), float((1 - self.abar[tau]).sqrt())

    def x0_pred(self, z, t, emb):
        self.nfe += 1
        a, _ = self.ab(t)
        return self.mu + self.w * (z / a - self.mu)


class FakeTwoPromptModel(FakeDiagonalModel):
    """Same, but the branch is chosen by ``emb`` so ``reward.lr_map`` can be exercised."""

    def __init__(self, mu_src, mu_tar, w, abar=0.0125):
        super().__init__(mu_src, w, abar=abar)
        self.mu_tar = mu_tar

    def x0_pred(self, z, t, emb):
        self.nfe += 1
        a, _ = self.ab(t)
        mu = self.mu if float(emb.reshape(-1)[0]) == 0.0 else self.mu_tar
        return mu + self.w * (z / a - mu)


# --------------------------------------------------------------------------- #
# T-K1 .. T-K4  (masks.py)
# --------------------------------------------------------------------------- #
def test_t_k1_pure_noise_gets_no_weight():
    gen = torch.Generator().manual_seed(0)
    means = []
    for _ in range(50):
        Rs = torch.randn(4, 4, 64, 64, generator=gen)
        w, T = masks.snr_weight(Rs, c=1.0, pool=3)
        assert float(T.mean()) > 0.5
        means.append(float(w.mean()))
    assert float(np.mean(means)) < 0.1, np.mean(means)
    assert float(np.mean(means)) > 0.0


def test_t_k2_strong_signal_is_kept():
    gen = torch.Generator().manual_seed(0)
    means = []
    for _ in range(50):
        mu = torch.randn(4, 1, 1, generator=gen)
        mu = mu / mu.norm() * float(np.sqrt(10.0))       # ||mu_p||^2 = 10 over the 4 channels
        Rs = mu[None] + torch.randn(4, 4, 64, 64, generator=gen)
        w, _ = masks.snr_weight(Rs, c=1.0, pool=3)
        means.append(float(w.mean()))
    assert float(np.mean(means)) > 0.85, np.mean(means)


def test_t_k3_full_mask_reproduces_the_plain_edit_and_n1_raises():
    gen = torch.Generator().manual_seed(2)
    Rs = torch.randn(4, 4, 64, 64, generator=gen)
    R0 = Rs[0]
    full = torch.ones(64, 64)
    oracle = full.unsqueeze(0).unsqueeze(0) * R0
    plain = 1.0 * R0
    assert torch.allclose(oracle[0], plain, atol=1e-7)
    with pytest.raises(ValueError):
        masks.snr_weight(Rs[:1], c=1.0, pool=3)


def test_t_k4_auroc_and_iou():
    gt = np.zeros((64, 64), np.float32)
    gt[16:48, 16:48] = 1.0
    assert abs(masks.auroc(gt, gt) - 1.0) < 1e-9
    assert abs(masks.iou(gt, gt) - 1.0) < 1e-9
    rng = np.random.default_rng(0)
    random_scores = rng.random((64, 64)).astype(np.float32)
    assert abs(masks.auroc(random_scores, gt) - 0.5) < 0.02
    # sqrt(0.1) ~ 0.32: about 32% of the pixels survive the 0.5 threshold of a uniform map
    random_iou = masks.iou(random_scores, gt)
    assert 0.0 < random_iou < 0.3, random_iou


def test_t_k4b_mask_quality_rejects_empty_gt():
    empty = np.zeros((64, 64), np.float32)
    assert np.isnan(masks.auroc(empty, empty))
    assert np.isnan(masks.iou(empty, empty))


# --------------------------------------------------------------------------- #
# T-N1 .. T-N3  (noise_ops.py)
# --------------------------------------------------------------------------- #
def test_t_n1_fixed_point_error_is_w_power_j():
    gen = torch.Generator().manual_seed(5)
    w = torch.linspace(0.2, 0.9, 4).view(1, 4, 1, 1)
    mu = torch.randn(1, 4, 1, 1, generator=gen)
    x_s = torch.randn(1, 4, 8, 8, generator=gen)
    eps0 = torch.randn(1, 4, 8, 8, generator=gen)
    model = FakeDiagonalModel(mu, w)
    a, b = model.ab(0.9)
    eps_star = (a / b) * (x_s - mu) / w * (1.0 - w)        # analytic fixed point

    for j_max in (1, 3, 5):
        eps_list, diag = noise_ops.fixed_point_eps(model, x_s, 0.9, torch.zeros(1, 1, 1), eps0, j_max=j_max)
        assert len(eps_list) == j_max + 1 and len(diag) == j_max + 1
        err = eps_list[-1] - eps_star
        expected = (1.0 - w) ** j_max * (eps0 - eps_star)
        rel = float((err - expected).abs().max() / expected.abs().max())
        assert rel < 1e-6, f"J={j_max} rel={rel}"


def test_t_n2_antithetic_is_minus_eps():
    eps = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(1))
    assert torch.equal(noise_ops.antithetic(eps), -eps)


@pytest.mark.gpu
def test_t_n3_dds_identity_on_the_real_model():
    """``A - R - e_src == 0`` to numerical precision (needs the real weights)."""

    from geoedit.data import PIEDataset
    from geoedit.edit_ops import make_eps
    from geoedit.model import SDTurbo

    dataset = PIEDataset(cache_dir=None)
    model = SDTurbo(device="cuda:0", unet_precision="fp32", vae_precision="fp32", text_precision="fp32")
    worst = 0.0
    for index in (0, 1, 2):
        sample = dataset[index]
        x_s = model.encode([sample.image()])
        eps = make_eps(sample.image_id, 0, model.device, model.unet_dtype)
        emb_src = model.embed([sample.src_prompt])
        emb_tar = model.embed([sample.tar_prompt])
        mu_src = source_mu(model, x_s, 0.9, emb_src, eps)
        R = residual_from_mu(model, x_s, 0.9, emb_tar, eps, mu_src)
        a, b = model.ab(0.9)
        mu_tar = model.x0_pred(a * x_s + b * eps, 0.9, emb_tar)
        A = mu_tar - x_s
        e_src = mu_src - x_s
        worst = max(worst, float((A - R - e_src).abs().max()))
    assert worst < 1e-5, worst


# --------------------------------------------------------------------------- #
# T-R1
# --------------------------------------------------------------------------- #
def test_t_r1_identical_prompts_give_a_zero_map():
    gen = torch.Generator().manual_seed(3)
    mu = torch.randn(1, 4, 1, 1, generator=gen)
    w = torch.full((1, 4, 1, 1), 0.5)
    model = FakeTwoPromptModel(mu, mu.clone(), w)
    x = torch.randn(1, 4, 8, 8, generator=gen)
    eps_list = [torch.randn(1, 4, 8, 8, generator=gen) for _ in range(2)]
    same = torch.zeros(1, 1, 1)
    l_map = reward.lr_map(model, x, 0.9, eps_list, same, same)
    assert float(l_map.abs().max()) < 1e-6


def test_t_r1b_different_prompts_give_the_right_sign():
    """``l > 0`` means "more target-like": ``eps_hat_src`` fits the noise *worse* than ``eps_hat_tar``.

    Derivation for the fake denoiser: ``eps_hat_c - eps = (I-W)(a/b)(x - mu_c) - W eps``, so on
    average ``l = 1/2 (||u_src||^2 - ||u_tar||^2)`` with ``u_c = (I-W)(a/b)(x - mu_c)``.  An
    image on the source manifold therefore has ``l < 0`` (the source branch explains it
    perfectly) and one on the target manifold has ``l > 0``.
    """
    gen = torch.Generator().manual_seed(4)
    mu = torch.randn(1, 4, 1, 1, generator=gen)
    w = torch.full((1, 4, 1, 1), 0.5)
    model = FakeTwoPromptModel(mu, mu + 3.0, w)
    eps_list = [torch.randn(1, 4, 8, 8, generator=gen) for _ in range(2)]
    emb_src, emb_tar = torch.zeros(1, 1, 1), torch.ones(1, 1, 1)
    on_target = reward.lr_map(model, mu + 3.0, 0.9, eps_list, emb_src, emb_tar)
    on_source = reward.lr_map(model, mu.clone(), 0.9, eps_list, emb_src, emb_tar)
    assert float(on_target.mean()) > 0.0
    assert float(on_source.mean()) < 0.0


# --------------------------------------------------------------------------- #
# T-S3  (mean-level curve matching)
# --------------------------------------------------------------------------- #
def _lines(n_img=30, seed=0, clip_shift=0.0, y_shift=0.0):
    rng = np.random.default_rng(seed)
    lam = np.arange(6, dtype=np.float64)
    clip = np.zeros((6, n_img))
    y = np.zeros((6, n_img))
    for i in range(6):
        clip[i] = lam[i] + clip_shift + rng.normal(0, 0.01, n_img)
        y[i] = 10.0 + lam[i] + y_shift + rng.normal(0, 0.01, n_img)
    return {"clip": clip, "y": y}


def test_t_s3_curve_matching_and_out_of_range():
    curves = {
        "ref": _lines(clip_shift=0.0, y_shift=0.0),
        "better": _lines(clip_shift=0.0, y_shift=2.0),
        "far": _lines(clip_shift=5.0, y_shift=0.0),
    }
    out = stats.mean_curve_match(curves, "ref", anchor_index=[2], n_boot=200, rng=np.random.default_rng(0))
    assert out["n"] == 30
    assert abs(out["results"]["2"]["ref"].delta) < 1e-6
    # both curves sit on the same CLIP axis, so at matched CLIP the gain is exactly 2 dB
    assert abs(out["results"]["2"]["better"].delta - 2.0) < 0.05
    assert out["results"]["2"]["far"].out_of_range
    assert np.isnan(out["results"]["2"]["far"].delta)


def test_t_boot_mean_of_max():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, (6, 40))
    b = rng.normal(0.0, 0.1, (6, 40))
    a[2] += 3.0                                       # the max over lambda sits at index 2
    ci = stats.boot_mean_of_max(a, b, n_boot=200, rng=np.random.default_rng(0))
    assert abs(ci["point"] - 4.0) < 0.1
    assert ci["lo"] < ci["point"] < ci["hi"]


# --------------------------------------------------------------------------- #
# T-X3  (official CLIP / metric agreement)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_t_x3_metrics_match_the_official_evaluator():
    """Every metric whose official definition matches ours agrees to < 1e-4.

    ``clip_source_whole`` is **deliberately excluded**: the official
    ``clip_similarity_source_image`` is ``CLIP(source_image, source_prompt)`` -- a per-image
    *constant* that does not depend on the output -- while ours is ``CLIP(output, source
    prompt)``, which is the diagnostic the P3 doc asks for ("非官方, 仅作参考").  The second
    half of the test pins that difference down instead of hiding it.
    """
    import sys
    from pathlib import Path

    from PIL import Image

    from geoedit.data import PIEDataset
    from geoedit.metrics import PieMetrics

    root = Path(__file__).resolve().parents[1]
    pnp = root / "third_party" / "PnPInversion"
    for path in (str(pnp), str(pnp / "evaluation")):
        if path not in sys.path:
            sys.path.insert(0, path)

    metrics = PieMetrics(device="cuda:0")            # also installs the local-CLIP patch
    from evaluation.evaluate import calculate_metric  # noqa: E402  (after the patch)
    from geoedit.model import SDTurbo

    dataset = PIEDataset(cache_dir=None)
    official = [
        "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part", "ssim_unedit_part",
        "clip_similarity_target_image", "clip_similarity_target_image_edit_part",
        "structure_distance",
    ]
    ours = ["psnr_unedit", "lpips_unedit", "mse_unedit", "ssim_unedit",
            "clip_target_whole", "clip_target_edit", "struct_dist"]

    model = SDTurbo(device="cuda:0", unet_precision="fp32", vae_precision="fp32",
                    text_precision="fp32", vae_only=True)
    worst = {k: 0.0 for k in ours}
    source_const_gap = 0.0
    for index in range(20):
        sample = dataset[index]
        source = sample.image()
        recon = model.decode(model.encode([source]))[0]
        output = Image.fromarray(recon)
        mask = sample.mask512.astype(np.float32)[:, :, np.newaxis].repeat(3, axis=2)
        got = metrics.evaluate(np.asarray(source), recon, sample.mask512,
                               sample.src_prompt, sample.tar_prompt)
        for key, official_key in zip(ours, official):
            value = calculate_metric(metrics.calc, official_key, source, output, mask, mask,
                                     sample.src_prompt, sample.tar_prompt)
            worst[key] = max(worst[key], abs(float(value) - float(got[key])))

        # the official "source" CLIP ignores the output entirely
        a = calculate_metric(metrics.calc, "clip_similarity_source_image", source, output,
                             mask, mask, sample.src_prompt, sample.tar_prompt)
        black = Image.fromarray(np.zeros_like(recon))
        b = calculate_metric(metrics.calc, "clip_similarity_source_image", source, black,
                             mask, mask, sample.src_prompt, sample.tar_prompt)
        source_const_gap = max(source_const_gap, abs(float(a) - float(b)))

    for key, diff in worst.items():
        assert diff < 1e-4, f"{key}: max abs diff {diff:.2e}"
    assert source_const_gap < 1e-6, source_const_gap
