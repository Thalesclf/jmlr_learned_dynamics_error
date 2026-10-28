"""Kuramoto--Sivashinsky + FNO extension for the realized-error-amplification study.

Scientific purpose
------------------
This script is the second-system / second-architecture experiment for the
manuscript "From One-Step Errors to Long-Horizon Rollouts: Realized Error
Amplification in Learned Dynamics".

It deliberately mirrors the Lorenz--96 diagnostics while changing both the
underlying dynamics and the learned surrogate:

    Lorenz--96 + residual MLP  ->  Kuramoto--Sivashinsky + 1D FNO.

The primary diagnostic propagator is evaluated along the TRUE trajectory,
consistent with the forced-error derivation

    delta_{k+1} = J_k delta_k + e_k + r_k,
    J_k = D f_theta(x_k),  e_k = f_theta(x_k) - x_{k+1}.

This distinction matters: a propagator along the learned nominal rollout is a
valid object for adversarial/stress-test perturbations, but it is not the same
linearization used by the true-trajectory defect decomposition.

Dependencies: numpy, torch
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    table = {"float32": torch.float32, "float64": torch.float64}
    if name not in table:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(table)}")
    return table[name]


# ---------------------------------------------------------------------------
# Kuramoto--Sivashinsky solver: Fourier pseudospectral + ETDRK4
# ---------------------------------------------------------------------------


def _ks_etdrk4_coefficients(k: np.ndarray, dt: float, contour_points: int = 32):
    """ETDRK4 coefficients using the contour formula of Kassam--Trefethen.

    The PDE convention is
        u_t + u u_x + u_xx + u_xxxx = 0,
    so the Fourier-space linear operator is L(k) = k^2 - k^4.
    """
    L = k**2 - k**4
    E = np.exp(dt * L)
    E2 = np.exp(0.5 * dt * L)

    r = np.exp(1j * np.pi * (np.arange(1, contour_points + 1) - 0.5) / contour_points)
    LR = dt * L[:, None] + r[None, :]

    # Real parts remove tiny contour-quadrature imaginary roundoff.
    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(
        np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1)
    )
    f2 = dt * np.real(
        np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1)
    )
    f3 = dt * np.real(
        np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1)
    )
    return E, E2, Q, f1, f2, f3


def simulate_ks(
    n_grid: int = 64,
    length: float = 22.0,
    dt: float = 0.25,
    steps: int = 12000,
    burnin: int = 2000,
    seed: int = 0,
    sample_stride: int = 1,
) -> np.ndarray:
    """Simulate 1D periodic Kuramoto--Sivashinsky with ETDRK4.

    Returns an array with shape (steps + 1, n_grid) after burn-in and optional
    temporal subsampling.  Keeping steps+1 states makes one-step pairs exact.
    """
    if n_grid % 2 != 0:
        raise ValueError("n_grid must be even for the rFFT-compatible periodic grid")
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")

    rng = np.random.default_rng(seed)
    x = length * np.arange(n_grid) / n_grid

    # Smooth random initial condition with deterministic large-scale content.
    u0 = (
        0.8 * np.cos(2.0 * np.pi * x / length)
        + 0.35 * np.sin(4.0 * np.pi * x / length)
        + 0.05 * rng.standard_normal(n_grid)
    )
    v = np.fft.fft(u0)

    k = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=length / n_grid)
    E, E2, Q, f1, f2, f3 = _ks_etdrk4_coefficients(k, dt)
    g = -0.5j * k

    # 2/3-rule de-aliasing for the quadratic nonlinearity.  Without this,
    # long integrations at the modest resolutions used for the ML benchmark
    # can accumulate aliased high-wavenumber energy and eventually blow up.
    mode_number = np.fft.fftfreq(n_grid) * n_grid
    dealias = np.abs(mode_number) <= (n_grid / 3.0)
    v[~dealias] = 0.0

    def nonlinear(vhat: np.ndarray) -> np.ndarray:
        u = np.fft.ifft(vhat).real
        nv = g * np.fft.fft(u * u)
        nv[~dealias] = 0.0
        return nv

    total_internal = burnin + steps * sample_stride
    states: List[np.ndarray] = []

    for j in range(total_internal + 1):
        if j >= burnin and (j - burnin) % sample_stride == 0:
            states.append(np.fft.ifft(v).real.copy())
            if len(states) == steps + 1:
                break

        Nv = nonlinear(v)
        a = E2 * v + Q * Nv
        Na = nonlinear(a)
        b = E2 * v + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlinear(c)
        v = E * v + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        # Keep the pseudospectral state in the de-aliased real-valued subspace.
        # The real projection enforces conjugate symmetry to roundoff; without
        # it, tiny symmetry errors can grow over very long chaotic integrations.
        v[~dealias] = 0.0
        v = np.fft.fft(np.fft.ifft(v).real)

    out = np.asarray(states)
    if out.shape != (steps + 1, n_grid):
        raise RuntimeError(f"KS simulator produced shape {out.shape}, expected {(steps + 1, n_grid)}")
    if not np.isfinite(out).all():
        raise FloatingPointError("KS simulation produced non-finite values")
    return out


# ---------------------------------------------------------------------------
# Data normalization
# ---------------------------------------------------------------------------


@dataclass
class ScalarNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean


def make_ks_dataset(
    trajectory: np.ndarray,
    train_fraction: float,
    dtype: torch.dtype,
    device: torch.device,
):
    data = torch.as_tensor(trajectory, dtype=dtype, device=device)
    n_pairs = len(data) - 1
    n_train = int(train_fraction * n_pairs)
    if n_train < 2 or n_train >= n_pairs:
        raise ValueError("train_fraction leaves an invalid train/test split")

    # A global scalar normalization preserves the homogeneous spatial symmetry
    # of the KS equation better than independently normalizing grid locations.
    train_states = data[: n_train + 1]
    mean = train_states.mean()
    std = train_states.std().clamp_min(torch.finfo(dtype).eps)
    norm = ScalarNormalizer(mean=mean, std=std)
    z = norm.encode(data)

    # Keep a contiguous sequence for diagnostics, plus pair tensors for training.
    xtr, ytr = z[:n_train], z[1 : n_train + 1]
    test_sequence = z[n_train:]
    xte, yte = test_sequence[:-1], test_sequence[1:]
    return xtr, ytr, xte, yte, test_sequence, norm


# ---------------------------------------------------------------------------
# 1D Fourier Neural Operator
# ---------------------------------------------------------------------------


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int, dtype: torch.dtype):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        real = torch.randn(in_channels, out_channels, modes, dtype=dtype)
        imag = torch.randn(in_channels, out_channels, modes, dtype=dtype)
        self.weight = nn.Parameter(scale * torch.complex(real, imag).to(complex_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, n]
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        n_freq = x_ft.shape[-1]
        m = min(self.modes, n_freq)
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, n_freq,
            dtype=x_ft.dtype, device=x.device,
        )
        out_ft[..., :m] = torch.einsum(
            "bim,iom->bom", x_ft[..., :m], self.weight[..., :m]
        )
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNO1d(nn.Module):
    """Compact periodic 1D FNO one-step surrogate.

    The external state shape is [..., n_grid]. Internally it is lifted to
    `width` channels and propagated through Fourier + pointwise layers.
    """

    def __init__(
        self,
        n_grid: int,
        modes: int = 16,
        width: int = 32,
        depth: int = 4,
        projection_width: int = 64,
        dtype: torch.dtype = torch.float64,
        outer_residual: bool = False,
    ):
        super().__init__()
        if modes > n_grid // 2 + 1:
            raise ValueError("modes exceeds the number of available rFFT modes")
        self.n_grid = n_grid
        self.outer_residual = outer_residual
        self.lift = nn.Linear(1, width, dtype=dtype)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes, dtype=dtype) for _ in range(depth)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1, dtype=dtype) for _ in range(depth)]
        )
        self.proj1 = nn.Linear(width, projection_width, dtype=dtype)
        self.proj2 = nn.Linear(projection_width, 1, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[-1] != self.n_grid:
            raise ValueError(f"Expected [batch,{self.n_grid}] or [{self.n_grid}], got {tuple(x.shape)}")

        skip = x
        z = self.lift(x.unsqueeze(-1)).transpose(1, 2)  # [B,width,N]
        for spec, pw in zip(self.spectral, self.pointwise):
            z = F.gelu(spec(z) + pw(z))
        z = z.transpose(1, 2)
        z = F.gelu(self.proj1(z))
        z = self.proj2(z).squeeze(-1)
        if self.outer_residual:
            z = skip + z
        return z.squeeze(0) if squeeze else z


# ---------------------------------------------------------------------------
# Training and rollout
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[float] = []
    n = len(xtr)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=xtr.device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            pred = model(xtr[idx])
            loss = torch.mean((pred - ytr[idx]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        mse = total / n
        history.append(mse)
        print(f"epoch {epoch+1:03d}/{epochs:03d} train_mse={mse:.6e}")
    model.eval()
    return history


def rollout(model: nn.Module, x0: torch.Tensor, steps: int) -> torch.Tensor:
    xs = [x0]
    x = x0
    for _ in range(steps):
        x = model(x)
        xs.append(x)
    return torch.stack(xs)


# ---------------------------------------------------------------------------
# Jacobians, propagators, and diagnostics
# ---------------------------------------------------------------------------


def explicit_jacobian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    return torch.autograd.functional.jacobian(
        lambda z: model(z), x, create_graph=False, vectorize=True
    ).detach()


def jacobians_along_states(model: nn.Module, states: torch.Tensor) -> List[torch.Tensor]:
    """Return Df(states[k]) for each state in `states`."""
    return [explicit_jacobian(model, x) for x in states]


def full_propagator(jacobians: Sequence[torch.Tensor]) -> torch.Tensor:
    if not jacobians:
        raise ValueError("At least one Jacobian is required")
    dim = jacobians[0].shape[0]
    P = torch.eye(dim, dtype=jacobians[0].dtype, device=jacobians[0].device)
    for J in jacobians:
        P = J @ P
    return P


def local_metrics(J: torch.Tensor) -> Dict[str, float]:
    eig = torch.linalg.eigvals(J)
    return {
        "spectral_radius_J": float(eig.abs().max()),
        "sigma1_J": float(torch.linalg.svdvals(J)[0]),
        "fro_J": float(torch.linalg.matrix_norm(J, ord="fro")),
    }


def jvp_step(model: nn.Module, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _, jv = torch.func.jvp(model, (x,), (v,))
    return jv


def vjp_step(model: nn.Module, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    _, vjp_fn = torch.func.vjp(model, x)
    return vjp_fn(w)[0]


def phi_mv(model: nn.Module, path_states: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply product Df(x_{T-1})...Df(x_0) to v without forming matrices."""
    out = v
    for x in path_states:
        out = jvp_step(model, x, out)
    return out


def phiT_mv(model: nn.Module, path_states: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = w
    for x in reversed(path_states):
        out = vjp_step(model, x, out)
    return out


def leading_singular_matrix_free(
    model: nn.Module,
    path_states: torch.Tensor,
    iters: int = 40,
    tol: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    v = torch.randn_like(path_states[0])
    v = v / torch.linalg.vector_norm(v).clamp_min(1e-30)
    for _ in range(iters):
        y = phi_mv(model, path_states, v)
        z = phiT_mv(model, path_states, y)
        v_new = z / torch.linalg.vector_norm(z).clamp_min(1e-30)
        # Singular vectors are defined only up to sign.
        dist = torch.minimum(
            torch.linalg.vector_norm(v_new - v),
            torch.linalg.vector_norm(v_new + v),
        )
        v = v_new
        if float(dist) < tol:
            break
    sigma = torch.linalg.vector_norm(phi_mv(model, path_states, v))
    return sigma.detach(), v.detach()


@dataclass
class DiagnosticSummary:
    actual_final_error: float
    relative_rollout_error: float
    linear_predicted_error: float
    linear_to_actual_ratio: float
    linear_vector_error: float
    defect_amplification_bound: float
    defect_l2_sum: float
    propagated_defect_norm_sum: float
    effective_defect_gain: float
    defect_cancellation_ratio: float
    sigma1_phi: float
    Gmax: float
    spectral_radius_J: float
    sigma1_J: float
    fro_J: float


def error_propagation_from_true_states(
    model: nn.Module,
    true_states: torch.Tensor,
    *,
    collect_full_spectrum: bool = True,
) -> Tuple[DiagnosticSummary, List[Dict], List[Dict]]:
    """Compute the paper's defect-propagation hierarchy on one true trajectory.

    true_states has shape [T+1, d] and is already normalized in the same
    coordinates used by the learned surrogate.
    """
    if true_states.ndim != 2 or len(true_states) < 2:
        raise ValueError("true_states must have shape [T+1,d]")
    T = len(true_states) - 1
    x0 = true_states[0]

    with torch.no_grad():
        model_states = rollout(model, x0, T)
        delta_actual = model_states[-1] - true_states[-1]
        actual = float(torch.linalg.vector_norm(delta_actual))
        denom = float(torch.linalg.vector_norm(true_states[-1])) + 1e-15
        defects = torch.stack([model(true_states[k]) - true_states[k + 1] for k in range(T)])

    Js = jacobians_along_states(model, true_states[:-1])
    Phi = full_propagator(Js)
    sigma1 = float(torch.linalg.svdvals(Phi)[0])
    local = local_metrics(Js[0])

    dim = x0.numel()
    P = torch.eye(dim, dtype=x0.dtype, device=x0.device)  # Phi(T,T) initially
    delta_linear = torch.zeros_like(x0)
    bound = 0.0
    terms: List[Dict] = []
    modes: List[Dict] = []

    for k in reversed(range(T)):
        e = defects[k]
        q = P @ e
        e_norm = float(torch.linalg.vector_norm(e))
        q_norm = float(torch.linalg.vector_norm(q))
        delta_linear = delta_linear + q

        if collect_full_spectrum:
            U, S, Vh = torch.linalg.svd(P)
        else:
            # We still need sigma_1 for the scalar worst-case defect bound. For
            # the small N used in the exact spectral campaign, full SVD is used.
            S = torch.linalg.svdvals(P)
            Vh = None
        sigma_tail = float(S[0])
        bound += sigma_tail * e_norm

        term: Dict[str, float | int] = {
            "k": k,
            "sigma_tail": sigma_tail,
            "defect_norm": e_norm,
            "propagated_defect_norm": q_norm,
            "defect_gain": q_norm / max(e_norm, 1e-30),
            "worst_case_term": sigma_tail * e_norm,
        }

        if collect_full_spectrum and e_norm > 1e-15 and q_norm > 1e-15:
            spectrum_nontrivial = float((S.max() - S.min()).abs()) > 1e-10
            coeffs = Vh @ e
            modal_energy = (S * coeffs).pow(2)
            modal_total = float(modal_energy.sum())
            if spectrum_nontrivial and modal_total > 1e-30:
                frac = (modal_energy / modal_total).detach().cpu().numpy()
                cumulative = np.cumsum(frac)
                m90 = int(np.searchsorted(cumulative, 0.90) + 1)
                participation = float(1.0 / np.sum(frac**2))
                lead_gap = float((S[0] - S[1]).abs()) if len(S) > 1 else math.inf
                alignment = (
                    float(torch.abs(torch.dot(Vh[0], e / e_norm))) if lead_gap > 1e-10 else math.nan
                )
                efficiency = q_norm / max(sigma_tail * e_norm, 1e-30)
                term.update(
                    tg_alignment=alignment,
                    gain_efficiency=efficiency,
                    realized_fraction_mode1=float(frac[0]),
                    realized_fraction_top3=float(np.sum(frac[: min(3, len(frac))])),
                    realized_fraction_top5=float(np.sum(frac[: min(5, len(frac))])),
                    modes_for_90pct=m90,
                    realized_participation_modes=participation,
                )
                for i in range(len(S)):
                    modes.append(
                        {
                            "k": k,
                            "mode": i + 1,
                            "sigma": float(S[i]),
                            "defect_coefficient_abs": float(torch.abs(coeffs[i])),
                            "realized_energy_fraction": float(frac[i]),
                            "cumulative_realized_fraction": float(cumulative[i]),
                        }
                    )
            else:
                term.update(
                    tg_alignment=math.nan,
                    gain_efficiency=q_norm / max(sigma_tail * e_norm, 1e-30),
                    realized_fraction_mode1=math.nan,
                    realized_fraction_top3=math.nan,
                    realized_fraction_top5=math.nan,
                    modes_for_90pct=-1,
                    realized_participation_modes=math.nan,
                )
        terms.append(term)
        P = P @ Js[k]

    terms.reverse()
    linear_norm = float(torch.linalg.vector_norm(delta_linear))
    propagated_sum = float(sum(float(t["propagated_defect_norm"]) for t in terms))
    defect_sum = float(sum(float(t["defect_norm"]) for t in terms))

    summary = DiagnosticSummary(
        actual_final_error=actual,
        relative_rollout_error=actual / denom,
        linear_predicted_error=linear_norm,
        linear_to_actual_ratio=linear_norm / (actual + 1e-15),
        linear_vector_error=float(torch.linalg.vector_norm(delta_linear - delta_actual)),
        defect_amplification_bound=bound,
        defect_l2_sum=defect_sum,
        propagated_defect_norm_sum=propagated_sum,
        effective_defect_gain=propagated_sum / (defect_sum + 1e-15),
        defect_cancellation_ratio=linear_norm / (propagated_sum + 1e-15),
        sigma1_phi=sigma1,
        Gmax=sigma1**2,
        **local,
    )
    return summary, terms, modes


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Dependency-free Spearman correlation (average ranks for ties)."""
    def rankdata(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a[order[j]] == a[order[i]]:
                j += 1
            ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
            i = j
        return ranks

    q = np.isfinite(x) & np.isfinite(y)
    x, y = x[q], y[q]
    if len(x) < 2:
        return math.nan
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(np.dot(rx, ry) / denom) if denom > 0 else math.nan


def write_rows_csv(rows: Iterable[Dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_campaign(
    model: nn.Module,
    test_sequence: torch.Tensor,
    horizons: Sequence[int],
    n_test_states: int,
    n_spectral_states: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    max_h = max(horizons)
    max_start = len(test_sequence) - max_h - 1
    if max_start < 0:
        raise ValueError("Test sequence is shorter than the maximum horizon")
    n_states = min(n_test_states, max_start + 1)
    indices = np.linspace(0, max_start, n_states, dtype=int)
    spectral_index_set = set(indices[np.linspace(0, n_states - 1, min(n_spectral_states, n_states), dtype=int)])

    rows: List[Dict] = []
    term_rows: List[Dict] = []
    mode_rows: List[Dict] = []

    for jj, idx in enumerate(indices):
        idx = int(idx)
        for T in horizons:
            true_states = test_sequence[idx : idx + T + 1]
            collect_full = idx in spectral_index_set
            summary, terms, modes = error_propagation_from_true_states(
                model, true_states, collect_full_spectrum=collect_full
            )
            row = {
                "system": "KS",
                "surrogate": "FNO1d",
                "seed": seed,
                "test_index": idx,
                "horizon": int(T),
                **asdict(summary),
            }
            rows.append(row)

            for term in terms:
                term_rows.append(
                    {
                        "system": "KS",
                        "surrogate": "FNO1d",
                        "seed": seed,
                        "test_index": idx,
                        "horizon": int(T),
                        **term,
                    }
                )
            for mode in modes:
                mode_rows.append(
                    {
                        "system": "KS",
                        "surrogate": "FNO1d",
                        "seed": seed,
                        "test_index": idx,
                        "horizon": int(T),
                        **mode,
                    }
                )
        print(f"diagnostic state {jj+1:03d}/{len(indices):03d}")

    return rows, term_rows, mode_rows


def summarize_campaign(rows: Sequence[Dict]) -> Dict:
    horizons = sorted({int(r["horizon"]) for r in rows})
    out: Dict[str, Dict] = {}
    for T in horizons:
        rr = [r for r in rows if int(r["horizon"]) == T]
        y = np.array([r["actual_final_error"] for r in rr], dtype=float)
        stats = {}
        for key in [
            "Gmax",
            "spectral_radius_J",
            "sigma1_J",
            "fro_J",
            "defect_l2_sum",
            "propagated_defect_norm_sum",
            "linear_predicted_error",
            "defect_amplification_bound",
        ]:
            x = np.array([r[key] for r in rr], dtype=float)
            stats[f"rho_{key}"] = spearman_corr(x, y)
        stats["median_Rlin"] = float(np.median([r["linear_to_actual_ratio"] for r in rr]))
        stats["median_cancellation"] = float(np.median([r["defect_cancellation_ratio"] for r in rr]))
        out[str(T)] = stats
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating Kuramoto--Sivashinsky trajectory...")
    traj = simulate_ks(
        n_grid=args.n_grid,
        length=args.length,
        dt=args.dt,
        steps=args.data_steps,
        burnin=args.burnin,
        seed=args.data_seed,
        sample_stride=args.sample_stride,
    )
    xtr, ytr, xte, yte, test_sequence, norm = make_ks_dataset(
        traj, train_fraction=args.train_fraction, dtype=dtype, device=device
    )

    model = FNO1d(
        n_grid=args.n_grid,
        modes=args.modes,
        width=args.width,
        depth=args.depth,
        projection_width=args.projection_width,
        dtype=dtype,
        outer_residual=args.outer_residual,
    ).to(device)

    print("Training FNO one-step surrogate...")
    history = train_model(
        model, xtr, ytr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    with torch.no_grad():
        test_mse = float(torch.mean((model(xte) - yte) ** 2))
    print(f"test one-step MSE = {test_mse:.6e}")

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    rows, terms, modes = diagnostic_campaign(
        model,
        test_sequence,
        horizons=horizons,
        n_test_states=args.n_test_states,
        n_spectral_states=args.n_spectral_states,
        seed=args.seed,
    )
    write_rows_csv(rows, outdir / "diagnostic_campaign.csv")
    write_rows_csv(terms, outdir / "defect_terms_campaign.csv")
    write_rows_csv(modes, outdir / "spectral_modes_campaign.csv")

    summary = summarize_campaign(rows)
    metadata = {
        "system": "Kuramoto-Sivashinsky",
        "surrogate": "FNO1d",
        "test_mse": test_mse,
        "training_history": history,
        "normalizer_mean": float(norm.mean),
        "normalizer_std": float(norm.std),
        "args": vars(args),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": str(device),
        "campaign_summary": summary,
    }
    with (outdir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalizer_mean": norm.mean.detach().cpu(),
            "normalizer_std": norm.std.detach().cpu(),
            "args": vars(args),
        },
        outdir / "checkpoint.pt",
    )

    print("\nCampaign summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs written to {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results_ks/seed_0")
    p.add_argument("--seed", type=int, default=0, help="training seed")
    p.add_argument("--data-seed", type=int, default=1729, help="shared KS trajectory seed")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")

    p.add_argument("--n-grid", type=int, default=64)
    p.add_argument("--length", type=float, default=22.0)
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--sample-stride", type=int, default=1)
    p.add_argument("--data-steps", type=int, default=12000)
    p.add_argument("--burnin", type=int, default=2000)
    p.add_argument("--train-fraction", type=float, default=0.7)

    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--projection-width", type=int, default=64)
    p.add_argument("--outer-residual", action="store_true")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument("--horizons", default="1,2,4,8,16,32,64")
    p.add_argument("--n-test-states", type=int, default=50)
    p.add_argument(
        "--n-spectral-states", type=int, default=15,
        help="subset of test states receiving exact full-spectrum diagnostics",
    )
    main(p.parse_args())
