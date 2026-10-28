"""Reference dynamical systems: Lorenz--96 (RK4) and Kuramoto--Sivashinsky (ETDRK4).

Both integrators reproduce the ones in ``experiments_campaign_spectral_jmlr.py``
and ``ks_fno_campaign.py``.  The L96 trajectories are bitwise identical to the
original campaign.  KS trajectories agree statistically but, being chaotic and
FFT based, they are not bitwise reproducible across platforms or library builds
(round-off differences grow at the leading Lyapunov rate).
"""
from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------
# Lorenz--96
# --------------------------------------------------------------------------

def lorenz96_rhs(x, forcing=8.0):
    return (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)) * np.roll(x, 1, axis=-1) - x + forcing


def l96_rk4_step(x, dt=0.01, forcing=8.0):
    k1 = lorenz96_rhs(x, forcing)
    k2 = lorenz96_rhs(x + 0.5 * dt * k1, forcing)
    k3 = lorenz96_rhs(x + 0.5 * dt * k2, forcing)
    k4 = lorenz96_rhs(x + dt * k3, forcing)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def simulate_l96(n=20, forcing=8.0, dt=0.01, steps=20000, burnin=2000, seed=0):
    """Same initial condition, burn-in and ordering as the original campaign."""
    rng = np.random.default_rng(seed)
    x = forcing * np.ones(n) + 0.01 * rng.standard_normal(n)
    traj = np.empty((steps, n))
    j = 0
    for k in range(steps + burnin):
        x = l96_rk4_step(x, dt, forcing)
        if k >= burnin:
            traj[j] = x
            j += 1
    return traj


def l96_rhs_jacobian(x, forcing=8.0):
    n = x.shape[-1]
    J = -np.eye(n)
    for i in range(n):
        ip1, im1, im2 = (i + 1) % n, (i - 1) % n, (i - 2) % n
        J[i, ip1] += x[im1]
        J[i, im2] -= x[im1]
        J[i, im1] += x[ip1] - x[im2]
    return J


def l96_rk4_jacobian(x, dt=0.01, forcing=8.0):
    """Exact Jacobian of one RK4 step (chain rule through the four stages)."""
    n = x.shape[-1]
    I = np.eye(n)
    k1 = lorenz96_rhs(x, forcing)
    K1 = l96_rhs_jacobian(x, forcing)
    x2 = x + 0.5 * dt * k1
    K2 = l96_rhs_jacobian(x2, forcing) @ (I + 0.5 * dt * K1)
    k2 = lorenz96_rhs(x2, forcing)
    x3 = x + 0.5 * dt * k2
    K3 = l96_rhs_jacobian(x3, forcing) @ (I + 0.5 * dt * K2)
    k3 = lorenz96_rhs(x3, forcing)
    x4 = x + dt * k3
    K4 = l96_rhs_jacobian(x4, forcing) @ (I + dt * K3)
    return I + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.0


# --------------------------------------------------------------------------
# Kuramoto--Sivashinsky, u_t + u u_x + u_xx + u_xxxx = 0 on [0, L), periodic
# --------------------------------------------------------------------------

def _etdrk4_coefficients(k, dt, contour_points=32):
    L = k ** 2 - k ** 4
    E = np.exp(dt * L)
    E2 = np.exp(0.5 * dt * L)
    r = np.exp(1j * np.pi * (np.arange(1, contour_points + 1) - 0.5) / contour_points)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1))
    return E, E2, Q, f1, f2, f3


class KSStepper:
    """De-aliased Fourier pseudospectral ETDRK4 step and its exact tangent map."""

    def __init__(self, n_grid=64, length=22.0, dt=0.25):
        self.n, self.length, self.dt = n_grid, length, dt
        self.k = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=length / n_grid)
        self.E, self.E2, self.Q, self.f1, self.f2, self.f3 = _etdrk4_coefficients(self.k, dt)
        self.g = -0.5j * self.k
        self.dealias = np.abs(np.fft.fftfreq(n_grid) * n_grid) <= n_grid / 3.0

    # nonlinear term N(v) = -(u^2/2)_x in Fourier space, and its derivative
    def _N(self, v):
        u = np.fft.ifft(v, axis=-1).real
        out = self.g * np.fft.fft(u * u, axis=-1)
        out[..., ~self.dealias] = 0.0
        return out

    def _dN(self, v, dv):
        u = np.fft.ifft(v, axis=-1).real
        du = np.fft.ifft(dv, axis=-1).real
        out = self.g * np.fft.fft(2.0 * u * du, axis=-1)
        out[..., ~self.dealias] = 0.0
        return out

    def _project(self, v):
        v = v.copy()
        v[..., ~self.dealias] = 0.0
        return np.fft.fft(np.fft.ifft(v, axis=-1).real, axis=-1)

    def step_hat(self, v):
        Nv = self._N(v)
        a = self.E2 * v + self.Q * Nv; Na = self._N(a)
        b = self.E2 * v + self.Q * Na; Nb = self._N(b)
        c = self.E2 * a + self.Q * (2.0 * Nb - Nv); Nc = self._N(c)
        v = self.E * v + self.f1 * Nv + 2.0 * self.f2 * (Na + Nb) + self.f3 * Nc
        return self._project(v)

    def step(self, u):
        v = np.fft.fft(u, axis=-1)
        v[..., ~self.dealias] = 0.0
        return np.fft.ifft(self.step_hat(v), axis=-1).real

    def jvp(self, u, du):
        """Exact tangent of ``step`` at state u (n,) for tangents du (m, n)."""
        v = np.fft.fft(u); v[~self.dealias] = 0.0
        dv = np.fft.fft(du, axis=-1); dv[..., ~self.dealias] = 0.0
        Nv = self._N(v); dNv = self._dN(v, dv)
        a = self.E2 * v + self.Q * Nv; da = self.E2 * dv + self.Q * dNv
        Na = self._N(a); dNa = self._dN(a, da)
        b = self.E2 * v + self.Q * Na; db = self.E2 * dv + self.Q * dNa
        Nb = self._N(b); dNb = self._dN(b, db)
        c = self.E2 * a + self.Q * (2.0 * Nb - Nv); dc = self.E2 * da + self.Q * (2.0 * dNb - dNv)
        dNc = self._dN(c, dc)
        dvn = self.E * dv + self.f1 * dNv + 2.0 * self.f2 * (dNa + dNb) + self.f3 * dNc
        dvn[..., ~self.dealias] = 0.0
        return np.fft.ifft(dvn, axis=-1).real

    def jacobian(self, u):
        return self.jvp(u, np.eye(self.n)).T


KS_TRAIN_MEAN = -0.0029783953383008543   # spatial mean of the training trajectory (data seed 1729)


def simulate_ks(n_grid=64, length=22.0, dt=0.25, steps=12000, burnin=2000, seed=0, sample_stride=1,
                mean=None):
    """Same initial condition, de-aliasing and burn-in as ``ks_fno_campaign.simulate_ks``.

    The spatial mean of u is an invariant of the KS equation, and a nonzero mean
    acts as a Galilean drift.  The random initial condition fixes this mean, so
    trajectories generated with different seeds belong to slightly different
    invariant sets.  Passing ``mean`` shifts the initial condition so that the
    trajectory has the prescribed mean (use KS_TRAIN_MEAN for evaluation data).
    """
    st = KSStepper(n_grid, length, dt)
    rng = np.random.default_rng(seed)
    x = length * np.arange(n_grid) / n_grid
    u0 = (0.8 * np.cos(2.0 * np.pi * x / length) + 0.35 * np.sin(4.0 * np.pi * x / length)
          + 0.05 * rng.standard_normal(n_grid))
    if mean is not None:
        u0 = u0 - u0.mean() + mean
    v = np.fft.fft(u0)
    v[~st.dealias] = 0.0
    states = []
    for j in range(burnin + steps * sample_stride + 1):
        if j >= burnin and (j - burnin) % sample_stride == 0:
            states.append(np.fft.ifft(v).real.copy())
            if len(states) == steps + 1:
                break
        v = st.step_hat(v)
    out = np.asarray(states)
    if not np.isfinite(out).all():
        raise FloatingPointError("KS simulation produced non-finite values")
    return out


# --------------------------------------------------------------------------
# Lyapunov exponents (Benettin/QR)
# --------------------------------------------------------------------------

def lyapunov_qr(step, jvp, x0, n_exp, n_steps, dt, transient=500, seed=0):
    """Leading ``n_exp`` Lyapunov exponents of a map x -> step(x).

    ``jvp(x, V)`` must return the tangent images of the rows of V (m, d).
    Returns exponents per unit physical time (divide by dt).
    """
    rng = np.random.default_rng(seed)
    x = np.array(x0, float)
    d = x.size
    Qm = np.linalg.qr(rng.standard_normal((d, n_exp)))[0]
    acc = np.zeros(n_exp)
    for it in range(transient + n_steps):
        W = jvp(x, Qm.T).T
        x = step(x)
        Qm, R = np.linalg.qr(W)
        s = np.sign(np.diag(R)); s[s == 0] = 1.0
        Qm = Qm * s
        if it >= transient:
            acc += np.log(np.abs(np.diag(R)))
    return acc / (n_steps * dt), x
