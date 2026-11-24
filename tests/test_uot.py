"""Unit tests for the partial-OT edit-region estimator.

Acceptance thresholds follow the task document (``2026_11_24.md`` section 4.2):
the windowed torch solver and the dense reference must agree to ``1e-4`` on ``sigma``,
and the entropic solution must approach the LP optimum with (almost) no mass on pairs
whose cost exceeds ``2 tau``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from geoedit.uot import (
    dense_cost,
    estimate_sigma,
    feature_scale,
    partial_ot_dense,
    partial_ot_window,
    window_cost,
    window_cost_transposed,
    window_index,
    window_radius,
)

DTYPE = torch.float64


def _random_features(n: int, d: int, seed: int = 0, drift: float = 0.3):
    gen = torch.Generator().manual_seed(seed)
    src = torch.randn(n, d, generator=gen, dtype=DTYPE)
    tgt = src + drift * torch.randn(n, d, generator=gen, dtype=DTYPE)
    return src, tgt


# --------------------------------------------------------------------------- #
# Geometry / cost
# --------------------------------------------------------------------------- #


def test_window_radius_never_truncates_reachable_radius():
    for tau in (0.25, 0.5, 1.0, 4.0, 16.0):
        for s_p in (1.0, 2.0, 3.0):
            r_max = s_p * math.sqrt(2.0 * tau)
            radius = window_radius(tau, s_p, max_radius=64)
            assert radius >= r_max, f"window R={radius} truncates r_max={r_max}"
            # default cap must not silently break the contract for sane settings
            if r_max <= 15:
                assert window_radius(tau, s_p) >= r_max


def test_window_cost_matches_dense():
    height = width = 7
    radius = 2
    feat_src, feat_tgt = _random_features(height * width, 3, seed=1)
    s_f, s_p = 1.3, 2.0

    win = window_index(height, width, radius)
    C = window_cost(feat_src, feat_tgt, win, s_f, s_p, chunk=3)
    Ct = window_cost_transposed(feat_src, feat_tgt, win, s_f, s_p, chunk=3)

    full = dense_cost(feat_src, feat_tgt, height, width, s_f, s_p, radius=radius)  # [src, tgt]
    expected = full.T.gather(1, win.idx)  # C[j, o] == full[src=j+o, tgt=j]
    assert torch.allclose(C[win.valid].double(), expected[win.valid].double(), atol=1e-12)
    assert torch.isinf(C[~win.valid]).all()

    expected_t = full.gather(1, win.idx)  # Ct[i, o] == full[src=i, tgt=i+o]
    assert torch.allclose(Ct[win.valid].double(), expected_t[win.valid].double(), atol=1e-12)
    assert torch.isinf(Ct[~win.valid]).all()


def test_transposed_cost_is_transpose():
    height = width = 9
    radius = 2
    feat_src, feat_tgt = _random_features(height * width, 5, seed=2)
    win = window_index(height, width, radius)
    C = window_cost(feat_src, feat_tgt, win, 1.0, 1.5)
    Ct = window_cost_transposed(feat_src, feat_tgt, win, 1.0, 1.5)
    # Ct[i, k] pairs (source i, target i+off[k]); C[j, k] pairs (target j, source j+off[k]).
    # The same pair appears in C at (target=i+off[k], offset=-off[k]), i.e. at the
    # reversed offset index, so Ct[i, k] == C[i+off[k], rev(k)].
    rev = torch.arange(win.n_offsets - 1, -1, -1).tolist()
    expected = torch.empty_like(Ct)
    for k, k_rev in enumerate(rev):
        expected[:, k] = C[win.idx[:, k], k_rev]

    mask = win.valid
    assert torch.allclose(Ct[mask].double(), expected[mask].double(), atol=1e-10)


# --------------------------------------------------------------------------- #
# Solver correctness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("height,radius", [(8, 7), (12, 2)])
def test_window_solver_matches_dense(height, radius):
    """Task-document acceptance: max |delta sigma| < 1e-4 between the two code paths."""
    n = height * height
    feat_src, feat_tgt = _random_features(n, 4, seed=3, drift=0.4)
    s_f = feature_scale(feat_src, feat_tgt)
    s_p, eps, tau = 1.5, 0.05, 1.0

    win = window_index(height, height, radius)
    C = window_cost(feat_src, feat_tgt, win, s_f, s_p)
    Ct = window_cost_transposed(feat_src, feat_tgt, win, s_f, s_p)
    sigma_win, *_ = partial_ot_window(C, Ct, win.idx, eps, tau, iters=20000, tol=1e-10)

    full = dense_cost(feat_src, feat_tgt, height, height, s_f, s_p, radius=radius)
    P, _ = partial_ot_dense(full, eps, tau, iters=20000, tol=1e-10)
    sigma_dense = P.sum(dim=0) * n  # (P^T 1)_j / b_j with b_j = 1/N

    diff = float((sigma_win - sigma_dense).abs().max().item())
    assert diff < 1e-4, f"sigma mismatch {diff:.3e}"


def test_entropic_solution_approaches_lp():
    """LP check: the linear objective must approach the LP value and put (almost) no
    mass on pairs whose cost exceeds ``2 tau``.

    The comparison is on the *linear* part ``<C - 2 tau, P>`` (as in the task document),
    not on the full entropic objective, whose ``eps * KL`` term adds an ``O(eps log N)``
    slack that has nothing to do with the threshold property.
    """
    import scipy.sparse as sp
    from scipy.optimize import linprog

    n, tau = 60, 1.0
    gen = torch.Generator().manual_seed(11)
    C = 3.0 * torch.rand(n, n, generator=gen, dtype=DTYPE)  # straddles 2*tau = 2

    # LP: min <C - 2 tau, P>  s.t.  P >= 0, P 1 <= a, P^T 1 <= b, a = b = 1/N
    cols = np.arange(n * n)
    A1 = sp.csr_matrix((np.ones(n * n), (np.repeat(np.arange(n), n), cols)), shape=(n, n * n))
    A2 = sp.csr_matrix((np.ones(n * n), (np.tile(np.arange(n), n), cols)), shape=(n, n * n))
    res = linprog(
        (C - 2.0 * tau).reshape(-1).numpy(),
        A_ub=sp.vstack([A1, A2]).tocsr(),
        b_ub=np.full(2 * n, 1.0 / n),
        bounds=(0, None),
        method="highs",
    )
    assert res.status == 0, res.message
    lp_linear = float(res.fun)

    gaps = {}
    for eps in (0.02, 0.01):
        P, _ = partial_ot_dense(C, eps, tau, iters=20000, tol=1e-13)
        linear = float(((C - 2.0 * tau) * P).sum().item())
        gaps[eps] = linear - lp_linear

        mass_outside = float(P[C > 2.0 * tau].sum().item())
        support_max_cost = float(C[P > 1e-12].max().item())
        assert mass_outside < 1e-10, f"eps={eps}: mass on pairs with C > 2 tau = {mass_outside:.3e}"
        assert support_max_cost <= 2.0 * tau + 1e-3
        # feasibility of the entropic solution, up to the soft-margin slack
        assert float(P.sum(1).max().item()) * n <= 1.0 + 1e-4
        assert float(P.sum(0).max().item()) * n <= 1.0 + 1e-3

    assert gaps[0.01] >= -1e-9, "entropic solution beat the LP optimum (infeasible?)"
    assert gaps[0.01] < 5e-3, f"entropic/LP linear gap {gaps[0.01]:.4e}"
    assert gaps[0.01] < gaps[0.02], f"gap did not shrink with eps: {gaps}"


def test_retained_mass_is_monotone_in_tau():
    n = 40
    feat_src, feat_tgt = _random_features(n, 3, seed=5, drift=0.6)
    s_f = feature_scale(feat_src, feat_tgt)
    full = dense_cost(feat_src, feat_tgt, 8, 5, s_f, 1.5, radius=None)

    masses = []
    for tau in (0.05, 0.25, 1.0, 4.0, 50.0):
        P, _ = partial_ot_dense(full, eps=0.02, tau=tau, iters=8000, tol=1e-12)
        masses.append(float(P.sum().item()))

    for prev, nxt in zip(masses, masses[1:]):
        assert nxt >= prev - 1e-12, f"retained mass not monotone: {masses}"
    assert masses[0] < 0.05, f"tau smallest should destroy almost everything, got {masses[0]}"
    assert masses[-1] > 0.95, f"tau largest should retain almost everything, got {masses[-1]}"


def test_radius_zero_is_pointwise_threshold():
    height = width = 8
    feat_src, feat_tgt = _random_features(height * width, 3, seed=7, drift=0.9)
    s_f = feature_scale(feat_src, feat_tgt)
    tau = 0.5

    eps = 0.002
    res = estimate_sigma(
        feat_src,
        feat_tgt,
        height,
        width,
        tau=tau,
        s_p=1.0,
        eps=eps,
        radius=0,
        iters=20000,
        tol=1e-12,
    )
    assert res.radius == 0

    diag = ((feat_src - feat_tgt) ** 2).sum(-1) / (s_f ** 2)
    # With K=1 the decision is "kept iff C_jj < 2 tau", but the entropic solution
    # transitions over a band of width ~ eps*log(N) around 2 tau; only judge the
    # tokens that sit clearly outside that band.
    n = height * width
    band = 10.0 * eps * math.log(n)
    should_edit = diag > 2.0 * tau + band
    should_keep = diag < 2.0 * tau - band
    assert int(should_edit.sum()) >= 3 and int(should_keep.sum()) >= 3, "degenerate instance"

    sigma = res.sigma
    assert float(sigma[should_edit].max().item()) < 1e-3, "pointwise threshold admits edit tokens"
    assert float(sigma[should_keep].min().item()) > 0.99, "pointwise threshold drops kept tokens"


def test_tau_endpoints_map_to_proposal_and_source():
    height = width = 10
    feat_src, feat_tgt = _random_features(height * width, 4, seed=9, drift=0.5)
    s_f = feature_scale(feat_src, feat_tgt)

    tiny = estimate_sigma(feat_src, feat_tgt, height, width, tau=1e-4, s_p=1.5, eps=0.05)
    assert float(tiny.sigma.max().item()) < 1e-2, "tau -> 0 must reproduce the proposal"

    huge = estimate_sigma(feat_src, feat_tgt, height, width, tau=1e3, s_p=1.5, eps=0.05, max_radius=64)
    assert float(huge.sigma.min().item()) > 0.99, "tau -> inf must reproduce the source"


# --------------------------------------------------------------------------- #
# Numerical robustness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("tau", [0.0, 0.5, 4.0])
def test_no_nan_under_extreme_scales(scale, tau):
    height = width = 11
    feat_src, feat_tgt = _random_features(height * width, 3, seed=13, drift=0.4)
    feat_src = feat_src * scale
    feat_tgt = feat_tgt * scale

    res = estimate_sigma(feat_src, feat_tgt, height, width, tau=tau, s_p=1.0, eps=0.1, radius=2)
    sigma = res.sigma
    assert torch.isfinite(sigma).all(), "sigma contains NaN/Inf"
    assert torch.isfinite(res.f).all() and torch.isfinite(res.g).all()
    assert float(sigma.min().item()) >= 0.0 and float(sigma.max().item()) <= 1.0
    assert float(res.f.max().item()) <= tau + 1e-9, "potentials must be clamped to tau"


def test_feature_scale_invariance():
    """Rescaling all features must not change sigma, because s_f rescales with them."""
    height = width = 9
    feat_src, feat_tgt = _random_features(height * width, 3, seed=17, drift=0.5)
    base = estimate_sigma(feat_src, feat_tgt, height, width, tau=1.0, s_p=1.5, eps=0.05)

    c = 7.3
    scaled = estimate_sigma(feat_src * c, feat_tgt * c, height, width, tau=1.0, s_p=1.5, eps=0.05)
    assert torch.allclose(scaled.s_f, base.s_f * c, rtol=1e-9)
    assert torch.allclose(base.sigma, scaled.sigma, atol=1e-9)


def test_edit_fraction_bounds():
    height = width = 12
    feat_src, feat_tgt = _random_features(height * width, 3, seed=19, drift=0.8)
    res = estimate_sigma(feat_src, feat_tgt, height, width, tau=1.0, s_p=1.0)
    assert 0.0 <= res.edit_fraction <= 1.0
