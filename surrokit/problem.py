"""Problem declaration for the surrokit engine.

Math-space contract: the engine sees only numbers. Every Y axis is
MAXIMIZED; axis 0 is the primary objective. Clients transform/negate on
their side and keep their units to themselves.
"""
from __future__ import annotations

from dataclasses import dataclass


class NotEnoughData(RuntimeError):
    """Fewer than 2 history rows: a GP cannot be fit."""


class InfeasibleError(RuntimeError):
    """constrained_max found zero feasible candidates even at k=0."""


@dataclass(frozen=True)
class Constraint:
    """Feasibility rule on one output axis: mean - k_sigma*sigma >= min."""
    axis: int
    min: float
    k_sigma: float = 1.0

    def __post_init__(self):
        if self.axis < 0:
            raise ValueError(f"Constraint.axis must be >= 0, got {self.axis}")
        if self.k_sigma < 0:
            raise ValueError(
                f"Constraint.k_sigma must be >= 0, got {self.k_sigma}")


@dataclass(frozen=True)
class Problem:
    """Search-space declaration: box bounds, integer dims, per-axis noise
    sigma (ABSOLUTE, in the client's Y-space; None = free MLL noise),
    optional budget Constraint (required by the constrained_max picker).
    """
    bounds_lo: tuple[float, ...]
    bounds_hi: tuple[float, ...]
    int_dims: tuple[int, ...] = ()
    noise: tuple[float, ...] | None = None
    constraint: Constraint | None = None

    def __post_init__(self):
        if len(self.bounds_lo) != len(self.bounds_hi):
            raise ValueError(
                f"bounds length mismatch: {len(self.bounds_lo)} lo vs "
                f"{len(self.bounds_hi)} hi")
        if not self.bounds_lo:
            raise ValueError("empty bounds")
        for i, (lo, hi) in enumerate(zip(self.bounds_lo, self.bounds_hi)):
            if not lo < hi:
                raise ValueError(f"dim {i}: lo {lo} must be < hi {hi}")
        for i in self.int_dims:
            if not 0 <= i < len(self.bounds_lo):
                raise ValueError(f"int_dim {i} out of range for "
                                 f"{len(self.bounds_lo)}D bounds")

    @property
    def dim(self) -> int:
        return len(self.bounds_lo)
