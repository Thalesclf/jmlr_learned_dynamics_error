"""Reference-free (prospective) error scores built from the model-trajectory propagator.

For a forecast started at x_0 the scores use only the surrogate, its free
rollout x_hat_k and the defect second moment Sigma estimated on independent
data (no future reference states):

    S_white = ( sum_k tr(Phi_k Sigma Phi_k^T) )^{1/2}          (uncorrelated defects)
    S_coh   = (tr Sigma)^{1/2} sum_k g_Sigma(Phi_k)            (persistent defects)
    Gmax_mdl = sigma_1^2( Df(x_hat_{T-1}) ... Df(x_hat_0) )

with Phi_k = Df(x_hat_{T-1}) ... Df(x_hat_{k+1}).  The scores are compared with
the realised error E_T through within-seed Spearman correlations.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_campaign_v2 import prepare
from realized_error.diagnostics import defect_second_moment, rollout


def scores(model, x0, T, Sigma, jac_cache=None):
    xs = rollout(model, x0, T)
    J = model.jacobians(xs[:-1])
    d = x0.size
    trS = np.trace(Sigma)
    P = np.eye(d)
    white = coh = 0.0
    for k in range(T - 1, -1, -1):
        t = float(np.einsum('ij,jk,ik->', P, Sigma, P))
        white += t
        coh += np.sqrt(t / trS)
        P = P @ J[k]
    s1 = np.linalg.svd(P, compute_uv=False)[0]
    return dict(S_white=np.sqrt(white), S_coh=np.sqrt(trS) * coh, Gmax_mdl=s1 ** 2), xs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["l96", "ks"], required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--campaign", required=True, help="campaign label with the same models (for E_T and windows)")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--horizons", default=None)
    p.add_argument("--n-states", type=int, default=100)
    a = p.parse_args()
    rows = []
    cache = {}
    for s in [int(v) for v in a.seeds.split(",")]:
        model, test, val, _ = prepare(a.system, a.models.format(seed=s), s, cache)
        Sigma = defect_second_moment(model, val)
        W = pd.read_csv(f"results/current/campaign/{a.campaign}/seed_{s}/windows.csv")
        hz = sorted(W['T'].unique()) if a.horizons is None else [int(h) for h in a.horizons.split(",")]
        starts = sorted(W.start.unique())[:a.n_states]
        for st in starts:
            for T in hz:
                sc, xs = scores(model, test[st], int(T), Sigma)
                E = float(np.linalg.norm(xs[-1] - test[st + T]))
                e0 = float(np.linalg.norm(model(test[st]) - test[st + 1]))
                rows.append(dict(seed=s, start=int(st), T=int(T), E=E, e0=e0, **sc))
        print(a.label, s, 'done', flush=True)
    df = pd.DataFrame(rows)
    out = Path("results/current/prospective"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{a.label}.csv", index=False)
    summ = []
    for T, g in df.groupby('T'):
        r = dict(T=T)
        for k in ['S_white', 'S_coh', 'Gmax_mdl', 'e0']:
            r['rho_' + k] = float(np.median([spearmanr(h[k], h.E).statistic for _, h in g.groupby('seed')]))
        r['rho_e0_x_Swhite'] = float(np.median([spearmanr(h.e0 * h.S_white, h.E).statistic for _, h in g.groupby('seed')]))
        r['median_ratio_Swhite_E'] = float(np.median(g.S_white / g.E)); r['median_ratio_Scoh_E'] = float(np.median(g.S_coh / g.E))
        summ.append(r)
    summ = pd.DataFrame(summ); summ.to_csv(out / f"{a.label}_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
