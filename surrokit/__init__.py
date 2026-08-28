"""surrokit — generic ask/tell GP surrogate engine."""
from .problem import Constraint, InfeasibleError, NotEnoughData, Problem

__version__ = "0.1.0"

__all__ = ["Constraint", "InfeasibleError", "NotEnoughData", "Problem",
           "__version__"]
