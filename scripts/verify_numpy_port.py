"""Recompute every archived L96 window of the original campaign with the NumPy port.

Compares rollout error, first-order reconstruction, B_T, Q_T, G_max, the local
Jacobian scalars and the test one-step MSE with results_l96_jmlr/.
"""
import json
import numpy as np
import pandas as pd
from run_campaign_v2 import prepare, l96_normalised_data
from realized_error.diagnostics import window_diagnostics

pairs = [('E', 'rollout_error'), ('Lin', 'linear_predicted_error'), ('B', 'defect_amplification_bound'),
         ('Sq', 'propagated_defect_norm_sum'), ('Gmax', 'Gmax'), ('Se', 'defect_l2_sum'),
         ('rhoJ', 'spectral_radius_J'), ('sigma1J', 'sigma1_J'), ('froJ', 'fro_J')]
worst = {a: 0.0 for a, _ in pairs}
n = 0
for s in range(5):
    m, test, _, _ = prepare('l96', f'results/seed_{s}/checkpoint.pt', s, {})
    J = m.jacobians(test)
    A = pd.read_csv(f'results_l96_jmlr/seed_{s}/diagnostic_campaign.csv')
    for _, r in A.iterrows():
        i, T = int(r.test_index), int(r.horizon)
        out, _ = window_diagnostics(m, test[i:i + T + 1], J[i:i + T], spectrum=False)
        for a, b in pairs:  # noqa
            worst[a] = max(worst[a], abs(out[a] / r[b] - 1))
        n += 1
    meta = json.load(open(f'results_l96_jmlr/seed_{s}/metadata.json'))
    _, meta_ck = __import__('realized_error.models', fromlist=['load_l96']).load_l96(f'results/seed_{s}/checkpoint.pt')
    z, ntr = l96_normalised_data(s, np.asarray(meta_ck['mean']), np.asarray(meta_ck['std']))
    mse = float(np.mean((m(z[ntr:-1]) - z[ntr + 1:]) ** 2))      # the archived (x_test, y_test) pairs
    print(f'seed {s}: test one-step MSE {mse:.16e} archived {meta["test_one_step_mse"]:.16e}', flush=True)
print(f'{n} windows; max relative difference per quantity:')
for a, v in worst.items():
    print(f'  {a:8s} {v:.2e}')
