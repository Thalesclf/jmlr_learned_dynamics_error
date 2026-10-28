"""Expansion of the learned tangent dynamics compared with the exact Lorenz--96 tangent map.

For the windows of a diagnostic campaign, compares log sigma_1 of the
reference-trajectory propagator of the surrogate, Phi(T, 0), with log sigma_1
of the exact RK4 tangent propagator along the same reference states, both in
the standardised coordinates of the surrogate (J_std = D^{-1} J_phys D with
D = diag(std)).  The output is the median over windows of the difference, per
seed and horizon.

python scripts/run_tangent_check.py --label l96_e2000 --models "results/current/l96_models/e2000/seed_{seed}/model.npz"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from realized_error.systems import l96_rk4_jacobian
from run_campaign_v2 import prepare


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--norm-from", default=None)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--horizons", default="20,40,80,160")
    p.add_argument("--n-states", type=int, default=100)
    a = p.parse_args()
    hz = [int(h) for h in a.horizons.split(",")]
    rows = []
    for s in [int(v) for v in a.seeds.split(",")]:
        nf = a.norm_from.format(seed=s) if a.norm_from else None
        model, test, _, norm = prepare('l96', a.models.format(seed=s), s, {}, nf)
        mean, std = np.asarray(norm['mean']), np.asarray(norm['std'])
        starts = np.linspace(0, len(test) - 320 - 1, a.n_states, dtype=int)     # same starts as the campaign
        for st in starts:
            Pm = np.eye(20); Pe = np.eye(20)
            for k in range(max(hz)):
                z = test[st + k]
                Pm = model.jacobian(z) @ Pm
                Je = l96_rk4_jacobian(z * std + mean)
                Pe = (Je * std[None, :] / std[:, None]) @ Pe
                if k + 1 in hz:
                    rows.append(dict(seed=s, start=int(st), T=k + 1,
                                     log_s1_model=float(np.log(np.linalg.svd(Pm, compute_uv=False)[0])),
                                     log_s1_exact=float(np.log(np.linalg.svd(Pe, compute_uv=False)[0]))))
    df = pd.DataFrame(rows)
    df['gap'] = df.log_s1_model - df.log_s1_exact
    out = Path("results/current/tangent"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{a.label}.csv", index=False)
    print(a.label)
    print(df.groupby('T')[['log_s1_model', 'log_s1_exact', 'gap']].median().round(3).to_string())
