"""Silence third-party console noise so an example's own output reads clean.

None of this is needed to USE surrokit -- the library itself never prints
(it logs to the "surrokit" logger). These are dependency artifacts:
linear_operator's Cholesky-jitter notices, torch's CUDA_HOME probe (a bare
print when torch.utils.cpp_extension is imported), and whatever
deprecation warnings the surrounding environment raises.
"""
from __future__ import annotations

import contextlib
import io
import warnings


def silence_third_party():
    warnings.filterwarnings("ignore")
    # cpp_extension prints "No CUDA runtime is found..." at import time;
    # import it now, under a redirect, so it cannot land mid-output later.
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            import torch.utils.cpp_extension  # noqa: F401
        except Exception:
            pass
