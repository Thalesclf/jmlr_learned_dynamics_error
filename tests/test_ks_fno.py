import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
torch = pytest.importorskip("torch")   # legacy PyTorch implementation

from ks_fno_campaign import (
    FNO1d,
    error_propagation_from_true_states,
    explicit_jacobian,
    simulate_ks,
)


def test_ks_long_integration_is_finite_and_mean_preserving():
    u = simulate_ks(
        n_grid=32, length=22.0, dt=0.25,
        steps=800, burnin=200, seed=7,
    )
    assert u.shape == (801, 32)
    assert np.isfinite(u).all()
    mean = u.mean(axis=1)
    assert np.max(np.abs(mean - mean[0])) < 1e-10


def test_fno_shape_and_jacobian():
    torch.set_default_dtype(torch.float64)
    model = FNO1d(n_grid=16, modes=4, width=4, depth=1,
                  projection_width=8, dtype=torch.float64)
    x = torch.randn(16)
    y = model(x)
    J = explicit_jacobian(model, x)
    assert y.shape == x.shape
    assert J.shape == (16, 16)
    assert torch.isfinite(J).all()


def test_one_step_defect_decomposition_is_exact():
    torch.set_default_dtype(torch.float64)
    model = FNO1d(n_grid=16, modes=4, width=4, depth=1,
                  projection_width=8, dtype=torch.float64)
    x0 = torch.randn(16)
    x1 = torch.randn(16)
    true_states = torch.stack([x0, x1])
    summary, _, _ = error_propagation_from_true_states(
        model, true_states, collect_full_spectrum=True
    )
    assert math.isclose(summary.linear_to_actual_ratio, 1.0, rel_tol=1e-10, abs_tol=1e-10)
    assert summary.linear_vector_error < 1e-10
