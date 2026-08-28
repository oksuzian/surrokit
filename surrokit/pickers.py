"""Pickers: acquisition strategies over a fitted GP.

All pickers return raw (q, d) tensors; ask() (Task 5) applies int-dim
rounding via sampling.emit_picks.
"""
from __future__ import annotations

import logging

import torch

from .problem import Constraint, InfeasibleError

log = logging.getLogger("surrokit")


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
