"""NumPy versions of the residual MLP (Lorenz--96) and the 1D FNO (KS).

The forward passes follow the PyTorch modules in the original scripts exactly
(``nn.Linear`` conventions, exact-erf GELU, ``rfft``/``irfft`` spectral layers),
so the archived ``.pt`` checkpoints give the same outputs to round-off.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from scipy.special import erf

from . import ptload

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


class ResidualMLP:
    """f(x) = x + W3 tanh(W2 tanh(W1 x + b1) + b2) + b3."""

    def __init__(self, sd):
        g = lambda k: np.asarray(sd[k], dtype=float)
        self.W1, self.b1 = g('net.0.weight'), g('net.0.bias')
        self.W2, self.b2 = g('net.2.weight'), g('net.2.bias')
        self.W3, self.b3 = g('net.4.weight'), g('net.4.bias')
        self.dim = self.W1.shape[1]

    def __call__(self, x):
        h1 = np.tanh(x @ self.W1.T + self.b1)
        h2 = np.tanh(h1 @ self.W2.T + self.b2)
        return x + h2 @ self.W3.T + self.b3

    def jacobians(self, X):
        """Df at every row of X: (B, d) -> (B, d, d)."""
        X = np.atleast_2d(X)
        h1 = np.tanh(X @ self.W1.T + self.b1)
        h2 = np.tanh(h1 @ self.W2.T + self.b2)
        A = (1.0 - h1 ** 2)[:, :, None] * self.W1[None]
        Bm = (1.0 - h2 ** 2)[:, :, None] * np.einsum('ij,bjd->bid', self.W2, A)
        return np.eye(self.dim)[None] + np.einsum('oi,bid->bod', self.W3, Bm)

    def jacobian(self, x):
        return self.jacobians(x[None])[0]

    def jvp(self, x, V):
        return V @ self.jacobian(x).T


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / SQRT2))


def gelu_prime(x):
    return 0.5 * (1.0 + erf(x / SQRT2)) + x * INV_SQRT_2PI * np.exp(-0.5 * x * x)


class FNO1d:
    """Lift -> depth x [GELU(SpectralConv + 1x1 Conv)] -> Linear-GELU-Linear."""

    def __init__(self, sd, n_grid=64, outer_residual=False):
        self.n = n_grid
        self.outer_residual = outer_residual
        self.lw = np.asarray(sd['lift.weight'], float)[:, 0]
        self.lb = np.asarray(sd['lift.bias'], float)
        self.depth = len([k for k in sd if k.startswith('spectral.')])
        self.SW = [np.asarray(sd[f'spectral.{i}.weight']) for i in range(self.depth)]
        self.PW = [np.asarray(sd[f'pointwise.{i}.weight'], float)[:, :, 0] for i in range(self.depth)]
        self.PB = [np.asarray(sd[f'pointwise.{i}.bias'], float) for i in range(self.depth)]
        self.P1W = np.asarray(sd['proj1.weight'], float); self.P1b = np.asarray(sd['proj1.bias'], float)
        self.P2W = np.asarray(sd['proj2.weight'], float); self.P2b = np.asarray(sd['proj2.bias'], float)
        self.modes = self.SW[0].shape[-1]
        self.dim = n_grid

    def _spec(self, z, W):
        zf = np.fft.rfft(z, axis=-1)
        m = min(self.modes, zf.shape[-1])
        out = np.zeros((z.shape[0], W.shape[1], zf.shape[-1]), dtype=np.complex128)
        out[..., :m] = np.einsum('bim,iom->bom', zf[..., :m], W[..., :m])
        return np.fft.irfft(out, n=z.shape[-1], axis=-1)

    def __call__(self, x, chunk=2048):
        squeeze = (x.ndim == 1)
        X = x[None] if squeeze else x
        outs = []
        for s in range(0, len(X), chunk):
            Xc = X[s:s + chunk]
            z = Xc[:, None, :] * self.lw[None, :, None] + self.lb[None, :, None]
            for i in range(self.depth):
                z = gelu(self._spec(z, self.SW[i]) + np.einsum('oi,bin->bon', self.PW[i], z)
                         + self.PB[i][None, :, None])
            z = np.transpose(z, (0, 2, 1))
            o = (gelu(z @ self.P1W.T + self.P1b) @ self.P2W.T + self.P2b)[..., 0]
            outs.append(Xc + o if self.outer_residual else o)
        out = np.concatenate(outs)
        return out[0] if squeeze else out

    def jvp(self, x, V):
        """Tangent images of the rows of V (m, n) at a single state x (n,)."""
        z = x[None, None, :] * self.lw[None, :, None] + self.lb[None, :, None]
        dz = V[:, None, :] * self.lw[None, :, None]
        for i in range(self.depth):
            pre = self._spec(z, self.SW[i]) + np.einsum('oi,bin->bon', self.PW[i], z) + self.PB[i][None, :, None]
            dpre = self._spec(dz, self.SW[i]) + np.einsum('oi,bin->bon', self.PW[i], dz)
            dz = gelu_prime(pre) * dpre
            z = gelu(pre)
        z = np.transpose(z, (0, 2, 1)); dz = np.transpose(dz, (0, 2, 1))
        dp = gelu_prime(z @ self.P1W.T + self.P1b) * (dz @ self.P1W.T)
        dout = (dp @ self.P2W.T)[..., 0]
        return V + dout if self.outer_residual else dout

    def jacobian(self, x):
        return self.jvp(x, np.eye(self.n)).T

    def jacobians(self, X, batch=16):
        """Batched Jacobians Df(x) for the rows of X: (B, n) -> (B, n, n)."""
        X = np.atleast_2d(X)
        n, out = self.n, []
        for s in range(0, len(X), batch):
            Xb = X[s:s + batch]
            B = len(Xb)
            z = Xb[:, None, :] * self.lw[None, :, None] + self.lb[None, :, None]           # B,w,N
            dz = np.broadcast_to(np.eye(n)[None, :, None, :] * self.lw[None, None, :, None],
                                 (B, n, len(self.lw), n)).copy()                          # B,tan,w,N
            for i in range(self.depth):
                pre = (self._spec(z, self.SW[i]) + np.einsum('oi,bin->bon', self.PW[i], z)
                       + self.PB[i][None, :, None])
                w = dz.shape[2]
                dpre = (self._spec(dz.reshape(B * n, w, n), self.SW[i]).reshape(B, n, -1, n)
                        + np.einsum('oi,btin->bton', self.PW[i], dz))
                dz = gelu_prime(pre)[:, None] * dpre
                z = gelu(pre)
            z = np.transpose(z, (0, 2, 1))                       # B,N,w
            dz = np.transpose(dz, (0, 1, 3, 2))                  # B,tan,N,w
            gp = gelu_prime(z @ self.P1W.T + self.P1b)           # B,N,p
            dout = ((gp[:, None] * (dz @ self.P1W.T)) @ self.P2W.T)[..., 0]   # B,tan,N
            Jb = np.transpose(dout, (0, 2, 1))
            if self.outer_residual:
                Jb = Jb + np.eye(n)[None]
            out.append(Jb)
        return np.concatenate(out)


# --------------------------------------------------------------------------
# checkpoint helpers
# --------------------------------------------------------------------------

def load_state(path):
    path = Path(path)
    if path.suffix == '.npz':
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}, {}
    raw = ptload.load(path)
    sd = raw['model_state_dict']
    meta = {k: v for k, v in raw.items() if k != 'model_state_dict'}
    return sd, meta


def load_l96(path):
    sd, meta = load_state(path)
    return ResidualMLP(sd), meta


def load_fno(path, n_grid=64):
    sd, meta = load_state(path)
    return FNO1d(sd, n_grid=n_grid), meta
