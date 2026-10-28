"""Revised diagnostic campaign (NumPy): factorisation, null models, Lyapunov-scaled horizons.

Examples
--------
python scripts/run_campaign_v2.py --system l96 --label l96_e20   --models "results/seed_{seed}/checkpoint.pt"
python scripts/run_campaign_v2.py --system l96 --label l96_e2000 --models "results/current/l96_models/e2000/seed_{seed}/model.npz"
python scripts/run_campaign_v2.py --system ks  --label ks_one_step --models "results_rollout_retry_final/ks/seed_{seed}/best_one_step.pt"

Outputs (results/current/campaign/<label>/seed_<s>/):
    windows.csv      one row per (initial state, horizon)
    terms.csv.gz     one row per (initial state, horizon, injection time)
    meta.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from realized_error.systems import simulate_l96, simulate_ks, KS_TRAIN_MEAN
from realized_error.models import load_l96, load_fno
from realized_error.diagnostics import window_diagnostics, defect_second_moment

L96_HORIZONS = "1,2,5,10,20,40,80,160,320"
KS_HORIZONS = "1,2,4,8,16,32,64,128,256"
KS_TEST_SEED = 314159       # independent KS diagnostic trajectory
KS_VAL_SEED = 271828        # independent KS trajectory for the defect second moment
L96_VAL_OFFSET = 1000       # L96 validation trajectory seed = model seed + offset


def l96_normalised_data(seed, mean, std, steps=20000, burnin=2000):
    traj = simulate_l96(seed=seed, steps=steps, burnin=burnin)
    ntr = int(0.7 * (len(traj) - 1))
    return (traj - mean) / std, ntr


def l96_normaliser(model_path, meta, norm_from=None):
    """Per-component normaliser of an L96 surrogate.

    Fine-tuned intervention checkpoints only store scalar summaries, so their
    normaliser is read from the pretrained checkpoint (``norm_from``).
    """
    if norm_from is not None:
        _, m2 = load_l96(norm_from)
        return np.asarray(m2['mean'], float), np.asarray(m2['std'], float)
    if 'mean' in meta and np.ndim(meta['mean']) == 1:
        return np.asarray(meta['mean'], float), np.asarray(meta['std'], float)
    d = np.load(model_path)
    return d['mean'], d['std']


def prepare(system, model_path, seed, cache, norm_from=None):
    if system == "l96":
        model, meta = load_l96(model_path)
        mean, std = l96_normaliser(model_path, meta, norm_from)
        z, ntr = l96_normalised_data(seed, mean, std)
        test = z[ntr:-1]                       # the held-out inputs used by the original campaign
        val = (simulate_l96(seed=seed + L96_VAL_OFFSET, steps=6000, burnin=2000) - mean) / std
        return model, test, val, dict(mean=mean.tolist(), std=std.tolist())
    model, meta = load_fno(model_path)
    mean = float(meta.get('normalizer_mean')); std = float(meta.get('normalizer_std'))
    if 'ks_test' not in cache:
        # evaluation trajectories share the conserved spatial mean of the training data
        cache['ks_test'] = simulate_ks(seed=KS_TEST_SEED, steps=10000, burnin=2000, mean=KS_TRAIN_MEAN)
        cache['ks_val'] = simulate_ks(seed=KS_VAL_SEED, steps=4000, burnin=2000, mean=KS_TRAIN_MEAN)
    return model, (cache['ks_test'] - mean) / std, (cache['ks_val'] - mean) / std, dict(mean=mean, std=std)


def run_seed(system, model_path, seed, horizons, n_states, outdir, cache, norm_from=None):
    t0 = time.time()
    model, test, val, norm = prepare(system, model_path, seed, cache, norm_from)
    Sigma, mu = defect_second_moment(model, val, return_mean=True)
    max_h = max(horizons)
    starts = np.linspace(0, len(test) - max_h - 1, n_states, dtype=int)
    need = np.unique(np.concatenate([np.arange(s, s + max_h) for s in starts]))
    jac = np.full((len(test), test.shape[1], test.shape[1]), np.nan)
    jac[need] = model.jacobians(test[need])
    t_jac = time.time() - t0
    rows, term_rows = [], []
    for s in starts:
        for T in horizons:
            ts = test[s:s + T + 1]
            out, terms = window_diagnostics(model, ts, jac[s:s + T], Sigma=Sigma, spectrum=True,
                                             mean_defect=mu)
            rows.append(dict(seed=seed, start=int(s), **out))
            for t in terms:
                term_rows.append(dict(seed=seed, start=int(s), T=T, **t))
    od = Path(outdir) / f"seed_{seed}"
    od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(od / "windows.csv", index=False)
    pd.DataFrame(term_rows).to_csv(od / "terms.csv.gz", index=False, float_format="%.7g")
    defects = model(test[:-1]) - test[1:]
    meta = dict(system=system, seed=seed, model=str(model_path), horizons=horizons, n_states=n_states,
                starts=starts.tolist(), test_len=len(test), normaliser=norm,
                test_one_step_mse=float(np.mean(defects ** 2)),
                persistence_mse=float(np.mean((test[1:] - test[:-1]) ** 2)),
                saturation_scale=float(np.sqrt(2.0 * np.trace(np.cov(test.T)))),
                defect_cov_trace=float(np.trace(Sigma)),
                defect_mean_sq_over_trace=float(mu @ mu / np.trace(Sigma)),
                defect_cov_eigs=np.linalg.eigvalsh(Sigma)[::-1].tolist(),
                seconds_jacobians=round(t_jac, 1), seconds_total=round(time.time() - t0, 1))
    (od / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{system} seed {seed}: one-step mse {meta['test_one_step_mse']:.3e}, {meta['seconds_total']}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["l96", "ks"], required=True)
    p.add_argument("--models", required=True, help="path pattern containing {seed}")
    p.add_argument("--label", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--horizons", default=None)
    p.add_argument("--n-states", type=int, default=100)
    p.add_argument("--outroot", default="results/current/campaign")
    p.add_argument("--norm-from", default=None, help="L96 only: pretrained checkpoint pattern holding the normaliser")
    a = p.parse_args()
    hz = [int(h) for h in (a.horizons or (L96_HORIZONS if a.system == "l96" else KS_HORIZONS)).split(",")]
    cache = {}
    for s in [int(v) for v in a.seeds.split(",")]:
        nf = a.norm_from.format(seed=s) if a.norm_from else None
        run_seed(a.system, a.models.format(seed=s), s, hz, a.n_states, Path(a.outroot) / a.label, cache, nf)
