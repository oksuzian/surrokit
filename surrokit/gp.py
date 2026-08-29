"""Fixed-noise SingleTaskGP fit + posterior predict.

float64 on CPU throughout: histories are small (hundreds of rows), CPU
beats GPU including transfer, and float64 keeps optimizer paths
deterministic. The library never touches torch's global default dtype.
"""
from __future__ import annotations

import logging

import torch

from .problem import NotEnoughData, Problem

log = logging.getLogger("surrokit")

DEVICE = torch.device("cpu")


def to_f64(a) -> torch.Tensor:
    return torch.as_tensor(a, dtype=torch.float64, device=DEVICE)


def bounds_tensor(problem: Problem) -> torch.Tensor:
    lo = torch.tensor(problem.bounds_lo, dtype=torch.float64, device=DEVICE)
    hi = torch.tensor(problem.bounds_hi, dtype=torch.float64, device=DEVICE)
    return torch.stack([lo, hi], dim=0)


def fit(problem: Problem, X, Y):
    """Fit a SingleTaskGP on (X, Y). Validates shapes against the Problem;
    raises NotEnoughData below 2 rows, ValueError on any mismatch."""
    Xt, Yt = to_f64(X), to_f64(Y)
    if Xt.ndim != 2 or Yt.ndim != 2:
        raise ValueError(f"X and Y must be 2D, got {Xt.ndim}D / {Yt.ndim}D")
    if Xt.shape[0] != Yt.shape[0]:
        raise ValueError(f"X has {Xt.shape[0]} rows but Y has {Yt.shape[0]}")
    if Xt.shape[1] != problem.dim:
        raise ValueError(f"X is {Xt.shape[1]}D but problem.dim is "
                         f"{problem.dim}")
    if problem.constraint is not None and problem.constraint.axis >= Yt.shape[1]:
        raise ValueError(f"constraint.axis {problem.constraint.axis} out of "
                         f"range for {Yt.shape[1]} output axes")
    if problem.noise is not None and len(problem.noise) != Yt.shape[1]:
        raise ValueError(f"noise has {len(problem.noise)} sigmas but Y has "
                         f"{Yt.shape[1]} output axes")
    if Xt.shape[0] < 2:
        raise NotEnoughData(f"{Xt.shape[0]} history rows; need >= 2 to fit")
    return _fit_model(Xt, Yt, bounds_tensor(problem), problem.noise)


def _fit_model(X, Y, bounds, noise):
    """Fit a SingleTaskGP (input Normalize + outcome Standardize).

    noise (ABSOLUTE per-output sigma) squares into train_Yvar -> a
    fixed-noise likelihood. Left free, MLL routinely over-fits noise on
    small replicated datasets and erases real optima; clients with
    replicate-measured noise should always pin it.
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    m = Y.shape[-1]
    train_Yvar = None
    if noise is not None:
        # Broadcast sigma^2 across rows: noise is a property of the
        # client's measurement budget, not the point.
        sig = torch.tensor([float(v) for v in noise],
                           dtype=Y.dtype, device=Y.device)
        train_Yvar = (sig ** 2).expand(Y.shape[0], m).contiguous()

    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        train_Yvar=train_Yvar,
        input_transform=Normalize(d=X.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=m),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    # Noise audit: likelihood noise is standardized (x stdvs = raw);
    # fixed-noise carries (m, n) -- collapse to per-output.
    noise_t = model.likelihood.noise.detach()
    noise_std = (noise_t.reshape(-1) if noise_t.numel() == m
                 else noise_t.reshape(m, -1)[:, 0]).sqrt()
    stdvs = model.outcome_transform.stdvs.detach().reshape(-1)
    raw = [f"{v:.3e}" for v in (noise_std * stdvs).tolist()]
    src = "FIXED (problem.noise)" if train_Yvar is not None else "MLL-fitted"
    log.info("GP noise sigma per output [%s]: raw=%s standardized=%s",
             src, raw, [f"{v:.3f}" for v in noise_std.tolist()])
    return model


def predict(model, X):
    """Posterior mean and stddev at X: two numpy arrays, each (n, m)."""
    Xq = to_f64(X)
    with torch.no_grad():
        post = model.posterior(Xq)
        mean = post.mean
        sig = post.variance.clamp_min(0).sqrt()
    return mean.cpu().numpy(), sig.cpu().numpy()
