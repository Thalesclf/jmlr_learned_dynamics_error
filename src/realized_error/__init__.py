"""Framework-free (NumPy) analysis of realised error amplification.

The original campaigns were run with PyTorch.  This package re-implements the
two surrogates, the reference solvers and every diagnostic used in the
manuscript in plain NumPy, so that the archived checkpoints can be analysed,
and the revised experiments rerun, without a deep-learning framework.
"""
__all__ = ["systems", "models", "diagnostics", "training", "ptload"]
