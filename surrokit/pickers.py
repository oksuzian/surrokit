"""Pickers: acquisition strategies over a fitted GP.

All pickers return raw (q, d) tensors; ask() (Task 5) applies int-dim
rounding via sampling.emit_picks.
"""
from __future__ import annotations

import logging

import torch

from .gp import _fit_model, bounds_tensor, to_f64
from .problem import Constraint, InfeasibleError, Problem
from .sampling import (ACQ_NUM_RESTARTS, ACQ_OPTIONS, ACQ_RAW_SAMPLES,
                       emit_picks, optimize_acq, sampler, sobol_cold_start)

log = logging.getLogger("surrokit")

PICKER_CHOICES = ("qnehvi", "qlnei", "qnparego", "hybrid", "constrained_max")


def _constrained_max(model, bounds, q: int, seed: int, pending,
                     constraint: Constraint, min_spacing: float = 0.10,
                     pool: int = 16384) -> torch.Tensor:
    """The q highest-axis-0 points the GP believes satisfy the constraint.

    Feasibility is mean[axis] - k_sigma*sigma[axis] >= min -- k-sigma,
    not mean-only, because a pick whose TRUE value lands past the
    threshold contributes nothing. The k-relaxation ladder (k -> k/2 ->
    0) trades margin for a full batch rather than returning fewer than q
    picks. pending rows seed the min-distance filter (NOT returned).
    """
    from scipy.stats import qmc

    ax = constraint.axis
    thr = float(constraint.min)
    k = float(constraint.k_sigma)

    d = bounds.shape[-1]
    unit = qmc.Sobol(d=d, scramble=True, seed=seed).random(pool)
    lo = bounds[0].cpu().numpy()
    hi = bounds[1].cpu().numpy()
    Xs = torch.tensor(lo + unit * (hi - lo), dtype=bounds.dtype,
                      device=bounds.device)
    with torch.no_grad():
        post = model.posterior(Xs)
        mean = post.mean
        std = post.variance.clamp_min(0).sqrt()
    obj = mean[:, 0]
    feas_margin = mean[:, ax] - k * std[:, ax]

    # Relax k rather than return an empty batch.
    used_k = k
    feasible = feas_margin >= thr
    for relaxed in (k * 0.5, 0.0):
        if int(feasible.sum()) >= q:
            break
        used_k = relaxed
        feasible = (mean[:, ax] - relaxed * std[:, ax]) >= thr
    n_feas = int(feasible.sum())
    if used_k != k:
        log.warning("constrained_max: only %d candidates at k=%ssigma; "
                    "relaxed to k=%ssigma (%d candidates)",
                    int((feas_margin >= thr).sum()), k, used_k, n_feas)
    if n_feas == 0:
        raise InfeasibleError(
            f"no candidate in the search box satisfies axis-{ax} >= "
            f"{thr:.6g} even at k=0; the threshold is wrong or the box "
            "has moved off the feasible region")

    idx_feas = torch.nonzero(feasible, as_tuple=False).squeeze(-1)
    order = idx_feas[torch.argsort(obj[idx_feas], descending=True)]
    norm = (Xs - bounds[0]) / (bounds[1] - bounds[0])
    avoid = []
    if pending is not None and len(pending):
        avoid = list((pending - bounds[0]) / (bounds[1] - bounds[0]))
    picks: list[int] = []
    for idx in order.tolist():
        if len(picks) >= q:
            break
        dmin = min((float((norm[idx] - a).pow(2).sum().sqrt())
                    for a in avoid), default=float("inf"))
        if dmin >= min_spacing:
            picks.append(idx)
            avoid.append(norm[idx])
    # Top up ONLY from the feasible set -- never leak infeasible picks.
    if len(picks) < q:
        for idx in order.tolist():
            if idx not in picks:
                picks.append(idx)
            if len(picks) >= q:
                break
    sel = torch.tensor(picks[:q])
    log.info("constrained_max: %d/%d candidates feasible at k=%ssigma "
             "(axis-%d >= %.6g); picked q=%d, predicted axis-0 "
             "%.3f-%.3f", n_feas, pool, used_k, ax, thr, len(sel),
             float(obj[sel].min()), float(obj[sel].max()))
    return Xs[sel].detach()


def _qnehvi(model, X, Y, bounds, q: int, seed: int, pending=None):
    """qLogNEHVI (log-stabilized qNEHVI, Ament 2023) for q candidates.

    pending: optional (k, d) in-flight rows; the acqf fantasizes over
    them so replacements don't re-pick a running point.
    """
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement,
    )

    # Ref point = observed nadir pushed out 10% of span; subtract the
    # offset (sign-robust -- "x 1.1" only works when nadir is negative).
    nadir = Y.min(dim=0).values
    span = (Y.max(dim=0).values - nadir).abs().clamp(min=1e-9)
    ref_point = (nadir - 0.1 * span).tolist()

    acq = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=X,
        sampler=sampler(seed),
        prune_baseline=True,
        X_pending=pending,
    )
    return optimize_acq(acq, bounds, q)


def _qlnei(model, X, bounds, q: int, seed: int, pending=None):
    """qLogNoisyExpectedImprovement over axis 0 only.

    botorch's single-objective MC acquisition functions refuse a bare
    multi-output model (UnsupportedError: "Must specify an objective or
    a posterior transform") -- they need to know which axis is "the"
    objective. A weights=[1, 0, ..., 0] posterior transform selects
    axis 0 and is a no-op for m=1 (single-output), so this never
    changes single-output behavior.
    """
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.objective import ScalarizedPosteriorTransform

    m = model.num_outputs
    transform = (ScalarizedPosteriorTransform(
        weights=torch.tensor([1.0] + [0.0] * (m - 1), dtype=X.dtype,
                             device=X.device))
        if m > 1 else None)

    acq = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=X,
        sampler=sampler(seed),
        prune_baseline=True,
        X_pending=pending,
        posterior_transform=transform,
    )
    return optimize_acq(acq, bounds, q)


def _qnparego(model, X, Y, bounds, q: int, seed: int, pending=None):
    """qNParEGO: qLogNEI over a fresh random Chebyshev scalarization per
    candidate -- fans the batch across the WHOLE front.

    Seed discipline: weights drawn inside ONE torch.manual_seed(seed)
    block -- DISTINCT per candidate, REPRODUCIBLE per seed.
    Sequential-greedy via a growing pending set; can't use the shared
    optimize_acq (per-candidate scalarization). pending rows are
    conditioned on but NOT returned.
    """
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.utils.multi_objective.scalarization import (
        get_chebyshev_scalarization,
    )
    from botorch.utils.sampling import sample_simplex
    from botorch.optim import optimize_acqf

    torch.manual_seed(seed)
    pend = [pending] if pending is not None else []
    picks = []
    for _ in range(q):
        w = sample_simplex(d=Y.shape[-1], n=1, dtype=Y.dtype).squeeze(0)
        obj = GenericMCObjective(get_chebyshev_scalarization(weights=w, Y=Y))
        acq = qLogNoisyExpectedImprovement(
            model=model, X_baseline=X, sampler=sampler(seed),
            objective=obj, prune_baseline=True,
            X_pending=torch.cat(pend) if pend else None,
        )
        cand, _ = optimize_acqf(
            acq_function=acq, bounds=bounds, q=1,
            num_restarts=ACQ_NUM_RESTARTS, raw_samples=ACQ_RAW_SAMPLES,
            options=dict(ACQ_OPTIONS),
        )
        pend.append(cand)
        picks.append(cand)
    return torch.cat(picks).detach()


def _hybrid(model, X, Y, bounds, q: int, seed: int, pending=None,
            hv_frac: float = 0.6):
    """One batch = hv_frac qnehvi + rest qnparego; parego conditions on
    the qnehvi picks via pending so the halves don't collide."""
    q_hv = min(q, max(0, round(hv_frac * q)))
    q_pe = q - q_hv
    if q_hv == 0:
        return _qnparego(model, X, Y, bounds, q=q, seed=seed,
                         pending=pending)
    hv_cands = _qnehvi(model, X, Y, bounds, q=q_hv, seed=seed,
                       pending=pending)
    pe_pending = (torch.cat([pending, hv_cands])
                  if pending is not None else hv_cands)
    if q_pe == 0:
        return hv_cands
    pe_cands = _qnparego(model, X, Y, bounds, q=q_pe, seed=seed,
                         pending=pe_pending)
    return torch.cat([hv_cands, pe_cands])


def ask(problem: Problem, X, Y, q: int = 5, picker: str = "hybrid",
        seed: int = 0, pending=None, min_spacing: float = 0.10,
        pool: int = 16384, hv_frac: float = 0.6) -> list:
    """STATELESS batch proposal: fits the GP internally on every call.

    seed is used verbatim in every RNG stream. n < 2 rows -> Sobol cold
    start (never an error). int_dims are rounded in the returned lists.
    """
    if picker not in PICKER_CHOICES:
        raise ValueError(f"unknown picker {picker!r}; choose from "
                         f"{PICKER_CHOICES}")
    # Stateless determinism: pin torch's global stream too -- ambient
    # state otherwise leaks into fit_gpytorch_mll and optimize_acqf's
    # initial-condition/retry draws, making identical calls diverge.
    torch.manual_seed(seed)
    bounds = bounds_tensor(problem)
    Xt = (to_f64(X) if len(X)
          else torch.empty((0, problem.dim), dtype=torch.float64))
    pend = None
    if pending:
        pend = to_f64([[float(v) for v in row] for row in pending])
        if pend.shape[-1] != problem.dim:
            raise ValueError(f"pending dim {pend.shape[-1]} != problem "
                             f"dim {problem.dim}")
    if Xt.shape[0] < 2:
        log.info("cold start: %d history rows < 2 -> Sobol draw "
                 "(q=%d, seed=%d)", Xt.shape[0], q, seed)
        return emit_picks(sobol_cold_start(bounds, q, seed),
                          problem.int_dims)
    Yt = to_f64(Y)
    if Yt.ndim != 2 or Yt.shape[0] != Xt.shape[0]:
        raise ValueError(f"Y must be 2D with {Xt.shape[0]} rows")
    m = Yt.shape[1]
    if picker in ("qnehvi", "qnparego", "hybrid") and m < 2:
        raise ValueError(f"picker {picker!r} requires m >= 2 output "
                         f"axes, got {m}")
    if picker == "constrained_max":
        c = problem.constraint
        if c is None:
            raise ValueError("constrained_max requires problem.constraint")
        if c.axis >= m:
            raise ValueError(f"constraint.axis {c.axis} out of range for "
                             f"{m} output axes")
    model = _fit_model(Xt, Yt, bounds, problem.noise)
    if picker == "qlnei":
        cands = _qlnei(model, Xt, bounds, q=q, seed=seed, pending=pend)
    elif picker == "constrained_max":
        cands = _constrained_max(model, bounds, q=q, seed=seed,
                                 pending=pend, constraint=problem.constraint,
                                 min_spacing=min_spacing, pool=pool)
    elif picker == "hybrid":
        cands = _hybrid(model, Xt, Yt, bounds, q=q, seed=seed,
                        pending=pend, hv_frac=hv_frac)
    elif picker == "qnparego":
        cands = _qnparego(model, Xt, Yt, bounds, q=q, seed=seed,
                          pending=pend)
    else:
        cands = _qnehvi(model, Xt, Yt, bounds, q=q, seed=seed,
                        pending=pend)
    return emit_picks(cands, problem.int_dims)
