"""Seeded RNG streams, shared acquisition-optimize call, Sobol cold start.

The seed is used VERBATIM in every stream. Clients that need a per-round
derivation (e.g. `42 ^ round_idx`) apply it on their side before calling.
"""
from __future__ import annotations

import torch

# Acquisition-optimization budget -- ONE tuning point for every picker.
ACQ_NUM_RESTARTS = 16
ACQ_RAW_SAMPLES = 512
ACQ_OPTIONS = {"batch_limit": 5, "maxiter": 200}


def sampler(seed: int):
    """The shared qMC sampler all acquisition pickers use."""
    from botorch.sampling.normal import SobolQMCNormalSampler
    return SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=seed)


def optimize_acq(acq, bounds, q: int) -> torch.Tensor:
    """Shared optimize_acqf call for all acquisition pickers; returns (q, d).

    sequential=True is REQUIRED: joint mode is a ~q*d-dim problem and is
    orders of magnitude slower at q ~ 10.
    """
    from botorch.optim import optimize_acqf
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=q,
        num_restarts=ACQ_NUM_RESTARTS,
        raw_samples=ACQ_RAW_SAMPLES,
        options=dict(ACQ_OPTIONS),
        sequential=True,
    )
    return candidates.detach()


def sobol_cold_start(bounds: torch.Tensor, q: int, seed: int) -> torch.Tensor:
    """Draw q Sobol points over `bounds` for the no-history batch."""
    from botorch.utils.sampling import draw_sobol_samples
    cands = draw_sobol_samples(bounds=bounds, n=1, q=q, seed=seed).squeeze(0)
    return cands.detach()


def emit_picks(cands: torch.Tensor, int_dims) -> list:
    """Cast a (q, d) tensor to native-typed lists (int_dims rounded).

    Native Python types only, so results survive any JSON/msgpack layer.
    """
    int_set = set(int_dims)
    out = []
    for row in cands.cpu().numpy().tolist():
        out.append([int(round(v)) if i in int_set else float(v)
                    for i, v in enumerate(row)])
    return out
