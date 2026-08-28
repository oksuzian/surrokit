"""surrokit — generic ask/tell GP surrogate engine."""
from .gp import fit, predict
from .problem import Constraint, InfeasibleError, NotEnoughData, Problem

__version__ = "0.1.0"

__all__ = ["Constraint", "InfeasibleError", "NotEnoughData", "Problem",
           "fit", "predict", "__version__"]
