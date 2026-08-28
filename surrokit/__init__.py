"""surrokit — generic ask/tell GP surrogate engine."""
from .gp import fit, predict
from .pickers import PICKER_CHOICES, ask
from .problem import Constraint, InfeasibleError, NotEnoughData, Problem

__version__ = "0.1.0"

__all__ = ["Constraint", "InfeasibleError", "NotEnoughData", "Problem",
           "PICKER_CHOICES", "ask", "fit", "predict", "__version__"]
