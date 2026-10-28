"""Manual-gradient training of the residual MLP (one-step and multistep rollout).

Only the Lorenz--96 surrogate is (re)trained in the revised study, so a small
hand-written reverse-mode implementation is sufficient.  Initialisation follows
``torch.nn.Linear`` (uniform on [-1/sqrt(fan_in), 1/sqrt(fan_in)]); the
optimiser is Adam with PyTorch defaults.  Gradients are verified against finite
differences in ``tests/test_numpy_core.py``.
"""
from __future__ import annotations
import numpy as np


def linear_init(rng, fan_out, fan_in):
    bound = 1.0 / np.sqrt(fan_in)
    return rng.uniform(-bound, bound, (fan_out, fan_in)), rng.uniform(-bound, bound, (fan_out,))


class MLPParams:
    """Parameters of f(x) = x + W3 tanh(W2 tanh(W1 x + b1) + b2) + b3."""

    def __init__(self, dim=20, hidden=128, seed=0, state_dict=None):
        if state_dict is not None:
            sd = state_dict
            self.p = {'W1': np.array(sd['net.0.weight'], float), 'b1': np.array(sd['net.0.bias'], float),
                      'W2': np.array(sd['net.2.weight'], float), 'b2': np.array(sd['net.2.bias'], float),
                      'W3': np.array(sd['net.4.weight'], float), 'b3': np.array(sd['net.4.bias'], float)}
        else:
            rng = np.random.default_rng(seed)
            W1, b1 = linear_init(rng, hidden, dim)
            W2, b2 = linear_init(rng, hidden, hidden)
            W3, b3 = linear_init(rng, dim, hidden)
            self.p = dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

    def copy(self):
        out = MLPParams.__new__(MLPParams)
        out.p = {k: v.copy() for k, v in self.p.items()}
        return out

    def state_dict(self):
        p = self.p
        return {'net.0.weight': p['W1'], 'net.0.bias': p['b1'], 'net.2.weight': p['W2'],
                'net.2.bias': p['b2'], 'net.4.weight': p['W3'], 'net.4.bias': p['b3']}

    def __call__(self, X):
        return self.forward(X)[0]

    def forward(self, X):
        p = self.p
        h1 = np.tanh(X @ p['W1'].T + p['b1'])
        h2 = np.tanh(h1 @ p['W2'].T + p['b2'])
        return X + h2 @ p['W3'].T + p['b3'], (X, h1, h2)

    def backward(self, cache, gY, need_input_grad=False):
        p = self.p
        X, h1, h2 = cache
        g = {'W3': gY.T @ h2, 'b3': gY.sum(0)}
        ga2 = (gY @ p['W3']) * (1.0 - h2 ** 2)
        g['W2'] = ga2.T @ h1; g['b2'] = ga2.sum(0)
        ga1 = (ga2 @ p['W2']) * (1.0 - h1 ** 2)
        g['W1'] = ga1.T @ X; g['b1'] = ga1.sum(0)
        if need_input_grad:
            return g, gY + ga1 @ p['W1']
        return g


def add_(acc, g, scale=1.0):
    for k in acc:
        acc[k] += scale * g[k]


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr, (self.b1, self.b2), self.eps, self.t = lr, betas, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            params[k] -= lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)


def clip_(grads, max_norm):
    tot = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))
    if max_norm is not None and tot > max_norm:
        s = max_norm / (tot + 1e-6)
        for k in grads:
            grads[k] *= s
    return tot


# ---------------------------------------------------------------- losses ----

def one_step_loss_and_grad(P, X, Y):
    """Mean over all entries of (f(X)-Y)^2, as torch's MSE(reduction='mean')."""
    Yh, cache = P.forward(X)
    r = Yh - Y
    return float(np.mean(r * r)), P.backward(cache, 2.0 * r / r.size)


def rollout_loss_and_grad(P, windows):
    """mean_{batch,dim} (f^H(x_0) - x_H)^2 with exact back-propagation through time.

    windows: (B, H+1, d) reference windows.
    """
    B, Hp1, d = windows.shape
    x = windows[:, 0]
    caches = []
    for _ in range(Hp1 - 1):
        x, c = P.forward(x)
        caches.append(c)
    r = x - windows[:, -1]
    loss = float(np.mean(r * r))
    gY = 2.0 * r / r.size
    grads = {k: np.zeros_like(v) for k, v in P.p.items()}
    for c in reversed(caches):
        g, gY = P.backward(c, gY, need_input_grad=True)
        add_(grads, g)
    return loss, grads


def windows_one_step(P, windows):
    B, Hp1, d = windows.shape
    return one_step_loss_and_grad(P, windows[:, :-1].reshape(-1, d), windows[:, 1:].reshape(-1, d))


# ------------------------------------------------------------- training ----

def train_one_step(P, xtr, ytr, epochs, batch_size=256, lr=1e-3, lr_final=None, seed=0,
                   xva=None, yva=None, log_every=0):
    """Mini-batch Adam with optional cosine decay of the learning rate."""
    rng = np.random.default_rng(10_000 + seed)
    opt = Adam(P.p, lr=lr)
    n = len(xtr)
    total = epochs * int(np.ceil(n / batch_size))
    step, hist = 0, []
    for ep in range(epochs):
        perm = rng.permutation(n)
        run = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss, g = one_step_loss_and_grad(P, xtr[idx], ytr[idx])
            run += loss * len(idx)
            if lr_final is None:
                cur = lr
            else:
                cur = lr_final + 0.5 * (lr - lr_final) * (1 + np.cos(np.pi * step / max(total - 1, 1)))
            opt.step(P.p, g, lr=cur)
            step += 1
        rec = {'epoch': ep + 1, 'train_mse': run / n}
        if xva is not None:
            rec['test_mse'] = float(np.mean((P(xva) - yva) ** 2))
        hist.append(rec)
        if log_every and (ep + 1) % log_every == 0:
            print(rec, flush=True)
    return hist
