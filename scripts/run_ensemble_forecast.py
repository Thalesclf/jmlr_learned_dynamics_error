"""Reference-free error estimates from ensemble disagreement.

The realized-error account of the manuscript is retrospective: the defects
e_k = f(x_k) - x_{k+1} need the future reference states.  This script tests
whether the same forced recursion becomes usable before the fact when the
defects are replaced by an estimate that needs no reference data.

The five surrogates of a family are trained independently.  For member s,
the leave-one-out ensemble mean of the other members, fbar_{-s}, is used as a
stand-in for the true map, and the defect proxy along the member's own free
rollout x_hat_k is

    e_hat_k = f_s(x_hat_k) - fbar_{-s}(x_hat_k).

Scores for a forecast started at x_0 (none uses states after x_0):

    S_hat   = sum_k ||e_hat_k||                         proxy defect magnitude
    L_hat   = ||d_T||, d_{k+1} = Df_s(x_hat_k) d_k + e_hat_k, d_0 = 0
                                                      first-order forced error, model trajectory
    D       = ||x_hat^s_T - mean_{j != s} x_hat^j_T||   nonlinear leave-one-out spread
    e0_hat  = ||e_hat_0||                               one-step proxy at the initial state

The scores are compared with the realized error E_T = ||x_hat^s_T - x_T||
through within-seed Spearman correlations, the median ratio score / E_T and the
AUROC for the 10% worst forecasts of each seed.  The quality of the defect proxy
is measured on the reference states (cosine and norm ratio between e_hat and e).

Examples
--------
python scripts/run_ensemble_forecast.py --system l96 --label l96_e2000 --models "results/current/l96_models/e2000/seed_{seed}/model.npz"
python scripts/run_ensemble_forecast.py --system ks  --label ks_pre    --models "results_ks/seed_{seed}/checkpoint.pt"
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_campaign_v2 import prepare, L96_HORIZONS, KS_HORIZONS


def auroc(score, target, frac=0.1):
    """AUROC of ``score`` for flagging the ``frac`` largest values of ``target``."""
    n = len(target)
    k = max(1, int(round(frac * n)))
    pos = np.zeros(n, bool)
    pos[np.argsort(target)[-k:]] = True
    r = pd.Series(score).rank().to_numpy()
    return float((r[pos].sum() - k * (k + 1) / 2) / (k * (n - k)))


class Member:
    """A surrogate expressed in the normalised coordinates of a host member."""

    def __init__(self, model, norm, host_norm):
        self.m = model
        self.a = np.asarray(norm['std']) / np.asarray(host_norm['std'])
        self.b = (np.asarray(norm['mean']) - np.asarray(host_norm['mean'])) / np.asarray(host_norm['std'])

    def to_own(self, z):
        return (z - self.b) / self.a

    def __call__(self, z):
        return self.m(self.to_own(z)) * self.a + self.b


def run_member(system, paths, s, horizons, n_states, cache):
    model, test, val, norm = prepare(system, paths[s], s, cache)
    others = []
    for j in range(len(paths)):
        if j == s:
            continue
        mj, _, _, nj = prepare(system, paths[j], j, cache) if system == 'l96' else prepare(system, paths[j], j, cache)
        others.append(Member(mj, nj, norm))
    fbar = lambda Z: np.mean([o(Z) for o in others], axis=0)
    # quality of the defect proxy on reference states
    X = test[:-1]
    e_true = model(X) - test[1:]
    e_hat = model(X) - fbar(X)
    cos = np.sum(e_true * e_hat, 1) / (np.linalg.norm(e_true, axis=1) * np.linalg.norm(e_hat, axis=1))
    proxy = dict(seed=s, median_cos=float(np.median(cos)), mean_cos=float(np.mean(cos)),
                 median_norm_ratio=float(np.median(np.linalg.norm(e_hat, axis=1) / np.linalg.norm(e_true, axis=1))),
                 rho_norms=float(spearmanr(np.linalg.norm(e_hat, axis=1), np.linalg.norm(e_true, axis=1)).statistic))
    Tm = max(horizons)
    starts = np.linspace(0, len(test) - Tm - 1, n_states, dtype=int)
    # batched free rollouts of the host and of every other member
    Z = test[starts].copy()
    traj = [Z.copy()]
    for _ in range(Tm):
        Z = model(Z)
        traj.append(Z.copy())
    traj = np.stack(traj, 1)                                   # (n, Tm+1, d)
    others_T = {}
    for o in others:
        Zo = o.to_own(test[starts])
        for h in range(1, Tm + 1):
            Zo = o.m(Zo)
            if h in horizons:
                others_T.setdefault(h, []).append(Zo * o.a + o.b)
    rows = []
    for i, st in enumerate(starts):
        xs = traj[i]
        eh = xs[1:] - fbar(xs[:-1])                            # proxy defects along the own rollout
        d = np.zeros(xs.shape[1])
        S = 0.0
        for k in range(Tm):
            d = model.jvp(xs[k], d[None])[0] + eh[k]
            S += float(np.linalg.norm(eh[k]))
            T = k + 1
            if T in horizons:
                E = float(np.linalg.norm(xs[T] - test[st + T]))
                D = float(np.linalg.norm(xs[T] - np.mean([v[i] for v in others_T[T]], axis=0)))
                rows.append(dict(seed=s, start=int(st), T=T, E=E, L_hat=float(np.linalg.norm(d)), S_hat=S,
                                 D=D, e0_hat=float(np.linalg.norm(eh[0])),
                                 e0_true=float(np.linalg.norm(model(test[st]) - test[st + 1]))))
    return rows, proxy


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["l96", "ks"], required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--horizons", default=None)
    p.add_argument("--n-states", type=int, default=100)
    a = p.parse_args()
    seeds = [int(v) for v in a.seeds.split(",")]
    hz = [int(h) for h in (a.horizons or (L96_HORIZONS if a.system == 'l96' else KS_HORIZONS)).split(",")]
    paths = [a.models.format(seed=s) for s in seeds]
    cache, rows, proxies = {}, [], []
    for s in seeds:
        t0 = time.time()
        r, pr = run_member(a.system, paths, s, hz, a.n_states, cache)
        rows += r; proxies.append(pr)
        print(a.label, s, f"{time.time() - t0:.0f}s", pr, flush=True)
    df = pd.DataFrame(rows)
    out = Path("results/current/ensemble"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{a.label}.csv", index=False)
    summ = []
    for T, g in df.groupby('T'):
        r = dict(T=T)
        for k in ['D', 'L_hat', 'S_hat', 'e0_hat', 'e0_true']:
            r['rho_' + k] = float(np.median([spearmanr(h[k], h.E).statistic for _, h in g.groupby('seed')]))
            r['auroc_' + k] = float(np.median([auroc(h[k].to_numpy(), h.E.to_numpy()) for _, h in g.groupby('seed')]))
        r['ratio_D_E'] = float(np.median(g.D / g.E)); r['ratio_L_hat_E'] = float(np.median(g.L_hat / g.E))
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / f"{a.label}_summary.csv", index=False)
    (out / f"{a.label}_proxy.json").write_text(json.dumps(proxies, indent=1))
    print(summ.round(3).to_string(index=False))
