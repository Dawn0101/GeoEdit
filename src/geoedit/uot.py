"""Partial (TV-unbalanced) optimal transport for edit-region estimation.

Grid convention
---------------
Both the source and the proposal live on the *same* ``H x W`` token grid, so token
``i = y*W + x`` has position ``(y, x)``.  Uniform masses ``a_i = b_j = 1/N``.

Problem
-------
::

    min_{P >= 0, P1 <= a, P^T1 <= b}  <C, P> + eps * KL(P | a x b)
                                      + tau * (|a| - |P|) + tau * (|b| - |P|)

``tau`` is the price of destroying one unit of source mass *and* the price of creating
one unit of target mass, hence the name partial / TV-unbalanced OT.  In the
zero-entropy limit the problem is the LP ``const + min <C - 2 tau, P>`` over the
partial-transport polytope, so a pair costs more than the "price of building it from
scratch" as soon as ``C_ij > 2 tau`` and therefore carries no mass.

Cost
----
Local window ``|o|_inf <= R``::

    C[j, o] = ||f_src[j + o] - f_tgt[j]||^2 / s_f^2 + ||o||^2 / s_p^2

The two terms have separate scales on purpose: ``s_f`` is calibrated to the *typical
same-position drift* (so ``tau`` is expressed in "drift units"), while ``s_p`` is the
spatial tolerance in grid cells.  The largest displacement that can ever be matched is
``r_max = s_p * sqrt(2 tau)``, which is why :func:`window_radius` derives ``R`` from
``(tau, s_p)`` instead of letting the caller pick it: if ``R < r_max`` the window, not
the price, becomes the binding constraint and ``tau`` stops meaning what it should.

Readout
-------
``sigma_j = (P^T 1)_j / b_j in [0, 1]``: the fraction of target token ``j`` that can be
explained by transporting nearby source content.  ``sigma = 1`` means "keep the source
here", ``sigma = 0`` means "accept the proposal here".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

__all__ = [
    "Window",
    "SigmaResult",
    "window_index",
    "window_radius",
    "feature_scale",
    "window_cost",
    "window_cost_transposed",
    "partial_ot_window",
    "partial_ot_dense",
    "dense_cost",
    "estimate_sigma",
    "sigma_to_grid",
]


# --------------------------------------------------------------------------- #
# Window geometry
# --------------------------------------------------------------------------- #


@dataclass
class Window:
    """Precomputed neighbourhood structure on an ``H x W`` grid.

    ``idx[j, k]`` is the flat token index of ``j + off[k]`` (clamped at the border);
    ``valid[j, k]`` says whether that neighbour is inside the image.  ``dpos[k]`` is
    ``||off[k]||^2`` in grid units, i.e. the numerator of the spatial cost.
    """

    radius: int
    idx: torch.Tensor  # [N, K] long
    valid: torch.Tensor  # [N, K] bool
    off: torch.Tensor  # [K, 2] float  (dy, dx)
    dpos: torch.Tensor  # [K] float    ||o||^2
    height: int
    width: int

    @property
    def n_tokens(self) -> int:
        return int(self.idx.shape[0])

    @property
    def n_offsets(self) -> int:
        return int(self.idx.shape[1])


def window_index(
    height: int,
    width: int,
    radius: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Window:
    """Build (and cache) the local window: all offsets ``o`` with ``|o|_inf <= radius``.

    The geometry only depends on ``(H, W, R)``, so with 700 images x 60 configs the
    same few windows are rebuilt thousands of times; the cache is bounded to keep
    memory flat.
    """
    key = (int(height), int(width), int(radius), str(device), str(dtype))
    hit = _WINDOW_CACHE.get(key)
    if hit is not None:
        return hit
    win = _build_window(height, width, radius, device, dtype)
    if len(_WINDOW_CACHE) >= _WINDOW_CACHE_MAX:
        _WINDOW_CACHE.pop(next(iter(_WINDOW_CACHE)))
    _WINDOW_CACHE[key] = win
    return win


def _build_window(
    height: int,
    width: int,
    radius: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Window:
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    device = torch.device("cpu") if device is None else torch.device(device)

    ys = torch.arange(height, device=device)
    xs = torch.arange(width, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1).to(torch.long)  # [N, 2]

    offs = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    off = torch.tensor(offs, device=device, dtype=dtype)  # [K, 2]

    ny = base[:, None, 0] + off[None, :, 0].long()
    nx = base[:, None, 1] + off[None, :, 1].long()
    valid = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
    idx = (ny.clamp(0, height - 1) * width + nx.clamp(0, width - 1)).reshape(-1, len(offs))

    return Window(
        radius=radius,
        idx=idx,
        valid=valid,
        off=off,
        dpos=(off ** 2).sum(-1),
        height=height,
        width=width,
    )


_WINDOW_CACHE: dict = {}
_WINDOW_CACHE_MAX = 32


def _auto_chunk(n_tokens: int, n_offsets: int, dim: int, dtype: torch.dtype, target_mb: int = 128) -> int:
    """Chunk size so that one ``[chunk, K, D]`` intermediate stays near ``target_mb``."""
    item = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 8
    per_token = max(1, n_offsets * max(1, dim) * item)
    return int(max(1, min(n_tokens, (target_mb * 2**20) // per_token)))


def window_radius(tau: float, s_p: float, margin: int = 1, max_radius: int = 16) -> int:
    """Smallest window radius that does not truncate the reachable radius.

    A pair can only carry mass when ``C_ij < 2 tau``; with a non-negative feature
    term this requires ``||o||^2 / s_p^2 < 2 tau``, i.e. ``||o|| < s_p sqrt(2 tau)``.
    We take ``ceil`` of that radius plus a safety ``margin`` so the window is never
    the binding constraint, and cap it at ``max_radius`` for cost reasons (the cap is
    recorded in the run config when it actually binds).
    """
    if s_p <= 0:
        raise ValueError(f"s_p must be > 0, got {s_p}")
    if tau <= 0:
        return int(max(0, margin))
    r_max = float(s_p) * math.sqrt(2.0 * float(tau))
    return int(min(max(0, math.ceil(r_max)) + int(margin), int(max_radius)))


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #


def _cost_matrix(
    feat_query: torch.Tensor,
    feat_ref: torch.Tensor,
    win: Window,
    s_f: float,
    s_p: float,
    chunk: Optional[int] = None,
) -> torch.Tensor:
    """``D[j, o] = ||feat_ref[j+o] - feat_query[j]||^2 / s_f^2 + ||o||^2 / s_p^2``.

    Chunked over query tokens.  Too small a chunk is catastrophically slow (one kernel
    launch per chunk: ``chunk=8`` on 64x64 costs 135 ms, ``chunk=512`` costs 7 ms), so
    the default sizes the chunk from a fixed memory budget instead of capping it.
    """
    if feat_query.shape != feat_ref.shape:
        raise ValueError(
            f"feature shapes must match, got {tuple(feat_query.shape)} vs {tuple(feat_ref.shape)}"
        )
    if feat_query.shape[0] != win.n_tokens:
        raise ValueError(
            f"feature has {feat_query.shape[0]} tokens but the grid has {win.n_tokens}"
        )

    idx = win.idx
    n_tokens = idx.shape[0]
    out = torch.empty((n_tokens, idx.shape[1]), dtype=feat_query.dtype, device=feat_query.device)

    inv_sf2 = 1.0 / float(s_f) ** 2
    spatial = win.dpos / float(s_p) ** 2
    if chunk is None:
        chunk = _auto_chunk(n_tokens, idx.shape[1], feat_query.shape[1], feat_query.dtype)
    chunk = max(1, int(chunk))

    for lo in range(0, n_tokens, chunk):
        hi = min(lo + chunk, n_tokens)
        gathered = feat_ref[idx[lo:hi]]  # [c, K, D]
        diff = gathered - feat_query[lo:hi, None, :]
        out[lo:hi] = (diff * diff).sum(-1) * inv_sf2 + spatial[None, :]

    return out.masked_fill(~win.valid, float("inf"))


def window_cost(
    feat_src: torch.Tensor,
    feat_tgt: torch.Tensor,
    win: Window,
    s_f: float,
    s_p: float,
    chunk: Optional[int] = None,
) -> torch.Tensor:
    """Target-major cost ``C[j, o] = ||f_src[j+o] - f_tgt[j]||^2 / s_f^2 + ||o||^2 / s_p^2``."""
    return _cost_matrix(feat_tgt, feat_src, win, s_f, s_p, chunk)


def window_cost_transposed(
    feat_src: torch.Tensor,
    feat_tgt: torch.Tensor,
    win: Window,
    s_f: float,
    s_p: float,
    chunk: Optional[int] = None,
) -> torch.Tensor:
    """Source-major cost ``Ct[i, o] = ||f_src[i] - f_tgt[i+o]||^2 / s_f^2 + ||o||^2 / s_p^2``.

    This is the transpose of :func:`window_cost` re-laid-out on the *source* grid.  We
    compute it directly instead of transposing the windowed matrix with a
    ``[N, K, K]`` gather, which would cost hundreds of MB for nothing.
    """
    return _cost_matrix(feat_src, feat_tgt, win, s_f, s_p, chunk)


def feature_scale(
    feat_src: torch.Tensor,
    feat_tgt: torch.Tensor,
    q: float = 0.5,
    floor: float = 1e-8,
) -> torch.Tensor:
    """``s_f := sqrt(quantile_q of ||f_src - f_tgt||^2)`` at identical positions.

    Calibrating on the median same-position distance means ``tau`` is expressed in
    units of "typical drift": ``2 tau = 1`` is exactly one median drift per token.
    """
    d = ((feat_src - feat_tgt) ** 2).sum(-1)
    if d.dtype not in (torch.float32, torch.float64):
        d = d.to(torch.float32)
    # torch.quantile has a size limit on the input; N = H*W <= 4096 in this project.
    return torch.quantile(d, q).clamp_min(floor).sqrt().to(feat_src.dtype)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


@dataclass
class SigmaResult:
    """Output of :func:`estimate_sigma`."""

    sigma: torch.Tensor  # [N] in [0, 1]
    f: torch.Tensor  # [N] source potentials (clamped to tau)
    g: torch.Tensor  # [N] target potentials (clamped to tau)
    s_f: torch.Tensor
    radius: int
    iters: int
    converged: bool

    @property
    def edit_fraction(self) -> float:
        """``mean(1 - sigma)``: continuous estimate of the edit area fraction."""
        return float((1.0 - self.sigma).mean().item())


@torch.no_grad()
def partial_ot_window(
    C: torch.Tensor,
    Ct: torch.Tensor,
    idx: torch.Tensor,
    eps: float,
    tau: float,
    iters: int = 300,
    tol: float = 1e-5,
    check_every: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
    """Entropic partial OT on a windowed grid via Sinkhorn with clamped potentials.

    The TV penalty ``tau (1 - s)`` has the conjugate ``max(f, -tau)``, so the dual
    constraint is ``f_i <= tau`` (only the upper bound is active because the deficit
    is non-negative) and the proximal step is a truncation of the Sinkhorn potential
    to ``tau``.  Every half-iteration is therefore "Sinkhorn update + clamp", which is
    a monotone descent on the dual and converges to the partial-OT fixed point.

    Parameters
    ----------
    C : [N, K]
        Target-major cost; ``C[j, o]`` pairs target ``j`` with source ``j + o``.
    Ct : [N, K]
        Source-major cost; ``Ct[i, o]`` pairs source ``i`` with target ``i + o``.
    idx : [N, K]
        Neighbour table from :func:`window_index`.

    Returns
    -------
    sigma [N], f [N], g [N], iterations, converged
    """
    if C.shape != Ct.shape:
        raise ValueError(f"C and Ct must have the same shape, got {tuple(C.shape)} / {tuple(Ct.shape)}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    n_tokens = int(C.shape[0])
    dtype = C.dtype
    device = C.device

    log_mass = -math.log(float(n_tokens))
    tau_t = torch.as_tensor(float(tau), dtype=dtype, device=device)

    f = torch.zeros(n_tokens, dtype=dtype, device=device)
    g = torch.zeros_like(f)

    n_iter = 0
    converged = False
    check_every = max(1, int(check_every))
    for it in range(int(iters)):
        g = torch.minimum(
            -float(eps) * torch.logsumexp((f[idx] - C) / float(eps) + log_mass, dim=1),
            tau_t,
        )
        f_new = torch.minimum(
            -float(eps) * torch.logsumexp((g[idx] - Ct) / float(eps) + log_mass, dim=1),
            tau_t,
        )
        n_iter = it + 1
        # The convergence probe is a GPU->CPU sync; doing it every iteration dominated the
        # runtime (~4 ms/iter on a shared GPU, vs ~0.2 ms of actual work).
        if n_iter % check_every == 0 or n_iter == int(iters):
            delta = float((f_new - f).abs().max().item())
            f = f_new
            if delta < float(tol):
                converged = True
                break
        else:
            f = f_new

    log_p = 2.0 * log_mass + (f[idx] + g[:, None] - C) / float(eps)  # [N, K], target-major
    sigma = torch.exp(torch.logsumexp(log_p, dim=1) - log_mass).clamp(0.0, 1.0)
    return sigma, f, g, n_iter, converged


@torch.no_grad()
def partial_ot_dense(
    C: torch.Tensor,
    eps: float,
    tau: float,
    iters: int = 2000,
    tol: float = 1e-9,
    return_plan: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Dense reference solver on a full ``N x N`` cost matrix (tests / tiny grids only).

    Same fixed-point equations as :func:`partial_ot_window`, just without the window,
    so the two implementations are independent code paths and can be cross-checked.
    """
    n_tokens = int(C.shape[0])
    dtype = C.dtype
    device = C.device
    log_mass = -math.log(float(n_tokens))

    f = torch.zeros(n_tokens, dtype=dtype, device=device)
    g = torch.zeros_like(f)
    tau_t = torch.as_tensor(float(tau), dtype=dtype, device=device)

    n_iter = 0
    for it in range(int(iters)):
        # C is indexed [source i, target j]; the g-update sums over sources (dim=0).
        g = torch.minimum(
            -float(eps) * torch.logsumexp((f[:, None] - C) / float(eps) + log_mass, dim=0),
            tau_t,
        )
        f_new = torch.minimum(
            -float(eps) * torch.logsumexp((g[None, :] - C) / float(eps) + log_mass, dim=1),
            tau_t,
        )
        delta = float((f_new - f).abs().max().item())
        f = f_new
        n_iter = it + 1
        if delta < float(tol):
            break

    if not return_plan:
        return f, n_iter
    P = torch.exp(2.0 * log_mass + (f[:, None] + g[None, :] - C) / float(eps))
    return P, n_iter


def dense_cost(
    feat_src: torch.Tensor,
    feat_tgt: torch.Tensor,
    height: int,
    width: int,
    s_f: float,
    s_p: float,
    radius: Optional[int] = None,
    chunk: int = 32,
) -> torch.Tensor:
    """Full ``[N, N]`` cost matrix, ``+inf`` outside the window (for tests and E1).

    Uses the *same* arithmetic as :func:`_cost_matrix` (explicit broadcast, squared
    difference summed over the feature axis) instead of ``torch.cdist``, so the
    windowed and the dense solvers see bit-comparable costs.  ``cdist`` differs by
    ~1e-5 in float64 and ``exp(./eps)`` amplifies that past any useful tolerance.
    """
    device = feat_src.device
    n_src, n_tgt = int(feat_src.shape[0]), int(feat_tgt.shape[0])
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)  # [N, 2]

    inv_sf2 = 1.0 / float(s_f) ** 2
    d_feat = torch.empty((n_src, n_tgt), dtype=feat_src.dtype, device=device)
    chunk = max(1, int(chunk))
    for lo in range(0, n_src, chunk):
        hi = min(lo + chunk, n_src)
        diff = feat_src[lo:hi, None, :] - feat_tgt[None, :, :]
        d_feat[lo:hi] = (diff * diff).sum(-1) * inv_sf2

    delta = pos[:, None, :] - pos[None, :, :]  # [N_src, N_tgt, 2]
    d_pos = (delta ** 2).sum(-1) / float(s_p) ** 2

    cost = d_feat + d_pos
    if radius is not None:
        reach = (delta.abs().amax(-1) <= int(radius))
        cost = cost.masked_fill(~reach, float("inf"))
    return cost


# --------------------------------------------------------------------------- #
# One-stop API
# --------------------------------------------------------------------------- #


@torch.no_grad()
def estimate_sigma(
    feat_src: torch.Tensor,
    feat_tgt: torch.Tensor,
    height: int,
    width: int,
    *,
    tau: float,
    s_p: float,
    eps: float = 0.1,
    radius: Optional[int] = None,
    iters: int = 300,
    tol: float = 1e-5,
    check_every: int = 20,
    s_f: Optional[torch.Tensor] = None,
    chunk: Optional[int] = None,
    max_radius: int = 16,
) -> SigmaResult:
    """Estimate ``sigma`` on one image.

    ``feat_src`` / ``feat_tgt`` are ``[N, D]`` features on the *same* grid.  Keep the
    tensors on the target device; the cost is computed chunked so ``D = 768`` is fine.

    Pass ``radius=0`` to recover the pointwise residual-threshold ablation: the
    problem then decouples per position and ``sigma_j = 1`` iff ``C_jj < 2 tau``.
    """
    if feat_src.dim() == 3 and feat_src.shape[0] == 1:
        feat_src = feat_src[0]
    if feat_tgt.dim() == 3 and feat_tgt.shape[0] == 1:
        feat_tgt = feat_tgt[0]
    if feat_src.dim() != 2 or feat_tgt.dim() != 2:
        raise ValueError("features must be [N, D]")
    n_tokens = int(height) * int(width)
    if feat_src.shape[0] != n_tokens or feat_tgt.shape[0] != n_tokens:
        raise ValueError(
            f"expected {n_tokens} tokens for a {height}x{width} grid, "
            f"got {feat_src.shape[0]} / {feat_tgt.shape[0]}"
        )

    s_f = feature_scale(feat_src, feat_tgt) if s_f is None else s_f
    radius = window_radius(tau, s_p, max_radius=max_radius) if radius is None else int(radius)

    win = window_index(height, width, radius, device=feat_src.device, dtype=feat_src.dtype)
    C = window_cost(feat_src, feat_tgt, win, s_f, s_p, chunk=chunk)
    Ct = window_cost_transposed(feat_src, feat_tgt, win, s_f, s_p, chunk=chunk)

    sigma, f, g, n_iter, conv = partial_ot_window(
        C, Ct, win.idx, eps, tau, iters=iters, tol=tol, check_every=check_every
    )
    return SigmaResult(
        sigma=sigma,
        f=f,
        g=g,
        s_f=torch.as_tensor(s_f, dtype=feat_src.dtype, device=feat_src.device),
        radius=radius,
        iters=n_iter,
        converged=conv,
    )


def sigma_to_grid(sigma: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """``[N]`` (or ``[B, N]``) -> ``[H, W]`` (or ``[B, 1, H, W]``)."""
    if sigma.dim() == 1:
        return sigma.reshape(height, width)
    if sigma.dim() == 2:
        return sigma.reshape(sigma.shape[0], 1, height, width)
    raise ValueError(f"unsupported sigma shape {tuple(sigma.shape)}")
