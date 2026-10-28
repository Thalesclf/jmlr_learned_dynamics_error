"""Model-trajectory stress test: leading singular direction of Phi_mdl vs finite-amplitude PGD (NumPy, L96).

Differences from the archived campaign (experiments_campaign_spectral.py):
PGD restarts are all random (no restart is initialised at the transient-growth
direction), and the amplitude range extends to eps = 1 so that the transition
out of the linear regime is resolved.  eps is the Euclidean norm of the
perturbation in standardised coordinates, in which every component has unit
variance; eps = 1 is therefore 1/sqrt(20) = 22% of the RMS size of the state
fluctuations about the mean (an RMS of 0.22 standard deviations per component).

    G_NL(eps, v) = ||f^T(x0 + eps v) - f^T(x0)||^2 / eps^2,   ||v|| = 1
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from realized_error.models import load_l96
from realized_error.training import MLPParams
from run_campaign_v2 import prepare


def final(P, x, T):
    for _ in range(T):
        x = P(x)
    return x


def grad_sep(P, x0, delta, T, target):
    """Gradient of ||f^T(x0+delta) - target||^2 with respect to delta (exact BPTT)."""
    x = (x0 + delta)[None]
    caches = []
    for _ in range(T):
        x, c = P.forward(x)
        caches.append(c)
    r = x[0] - target
    g = 2.0 * r[None]
    for c in reversed(caches):
        _, g = P.backward(c, g, need_input_grad=True)
    return float(r @ r), g[0]


def pgd(P, x0, T, eps, steps, restarts, rng):
    target = final(P, x0[None], T)[0]
    best, best_d = -np.inf, None
    for _ in range(restarts):
        d = rng.standard_normal(x0.size); d *= eps / np.linalg.norm(d)
        lr = 0.25 * eps                       # damped power-iteration-like step
        for _ in range(steps):
            _, g = grad_sep(P, x0, d, T, target)
            d = d + lr * g / max(np.linalg.norm(g), 1e-300)
            n = np.linalg.norm(d)
            if n > eps:
                d *= eps / n
        val, _ = grad_sep(P, x0, d, T, target)
        if val > best:
            best, best_d = val, d.copy()
    return best_d / np.linalg.norm(best_d), best / eps ** 2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--n-states", type=int, default=20)
    p.add_argument("--eps", default="1e-4,1e-3,1e-2,3e-2,1e-1,3e-1,1")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--random-dirs", type=int, default=20)
    a = p.parse_args()
    rows = []
    for s in [int(v) for v in a.seeds.split(",")]:
        m, test, _, _ = prepare('l96', a.models.format(seed=s), s, {})
        sd = {k: getattr(m, at) for k, at in [('net.0.weight', 'W1'), ('net.0.bias', 'b1'), ('net.2.weight', 'W2'),
                                              ('net.2.bias', 'b2'), ('net.4.weight', 'W3'), ('net.4.bias', 'b3')]}
        P = MLPParams(state_dict=sd)
        rng = np.random.default_rng(100 + s)
        starts = np.linspace(0, len(test) - a.T - 1, a.n_states, dtype=int)
        for st in starts:
            x0 = test[st]
            xs = [x0]
            for _ in range(a.T):
                xs.append(m(xs[-1]))
            Phi = np.eye(x0.size)
            for x in xs[:-1]:
                Phi = m.jacobian(x) @ Phi
            U, S, Vt = np.linalg.svd(Phi)
            target = xs[-1]
            gain = lambda v, e: float(np.sum((final(P, (x0 + e * v)[None], a.T)[0] - target) ** 2) / e ** 2)
            for e in [float(v) for v in a.eps.split(",")]:
                v_adv, g_pgd = pgd(P, x0, a.T, e, a.steps, a.restarts, rng)
                R = rng.standard_normal((a.random_dirs, x0.size)); R /= np.linalg.norm(R, axis=1, keepdims=True)
                g_rand = [gain(r, e) for r in R]
                rows.append(dict(seed=s, start=int(st), T=a.T, eps=e, sigma1_sq=float(S[0] ** 2),
                                 sigma2_sq=float(S[1] ** 2), G_TG=gain(Vt[0], e), G_v2=gain(Vt[1], e),
                                 G_PGD=g_pgd, G_rand_mean=float(np.mean(g_rand)), G_rand_max=float(np.max(g_rand)),
                                 alignment=float(abs(Vt[0] @ v_adv))))
        print(a.label, 'seed', s, 'done', flush=True)
    out = Path("results/current/pgd") / a.label
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pgd.csv", index=False)
    summ = df.groupby('eps').agg(med_align=('alignment', 'median'), q10_align=('alignment', lambda x: x.quantile(.1)),
                                 med_TG_norm=('G_TG', lambda x: np.median(x / df.loc[x.index, 'sigma1_sq'])),
                                 med_PGD_norm=('G_PGD', lambda x: np.median(x / df.loc[x.index, 'sigma1_sq'])),
                                 med_ratio=('G_TG', lambda x: np.median(x / df.loc[x.index, 'G_PGD'])),
                                 mean_TG=('G_TG', 'mean'), mean_v2=('G_v2', 'mean'), mean_rand=('G_rand_mean', 'mean'),
                                 mean_randmax=('G_rand_max', 'mean'), mean_PGD=('G_PGD', 'mean'))
    summ.to_csv(out / "summary.csv")
    print(summ.round(4).to_string())
