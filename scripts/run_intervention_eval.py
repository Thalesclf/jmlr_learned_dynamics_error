"""Re-evaluate matched one-step / rollout fine-tuned surrogates on long test trajectories.

The original confirmatory test used 17 windows from one 1,200-step trajectory
per system.  Here every matched pair is evaluated on a much longer independent
trajectory with non-overlapping windows, and the paired effects are reported
per seed with window-bootstrap intervals.  The relative RMSE follows the
original definition (scalar de-normalisation, ratio to the true physical state).

Example
-------
python scripts/run_intervention_eval.py --system ks  --root results_rollout_retry_final/ks  --label ks_retry
python scripts/run_intervention_eval.py --system l96 --root results_rollout_retry_final/l96 --label l96_e20_retry \
       --norm-from "results/seed_{seed}/checkpoint.pt"
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from realized_error.systems import simulate_l96, simulate_ks, KS_TRAIN_MEAN
from realized_error.models import load_l96, load_fno
from run_campaign_v2 import l96_normaliser

HORIZONS = (1, 4, 8, 16, 32, 64, 128)


def batched_errors(model, test, starts, horizons):
    """Normalised-coordinate error vectors for all windows at the requested horizons."""
    x = test[starts].copy()
    out = {}
    for h in range(1, max(horizons) + 1):
        x = model(x)
        if h in horizons:
            out[h] = (x - test[starts + h], test[starts + h], x.copy())
    return out


def evaluate(system, paths, seed, test_seed, steps, spacing, norm_from):
    loaders = {'l96': load_l96, 'ks': load_fno}
    models = {arm: loaders[system](p)[0] for arm, p in paths.items()}
    meta = loaders[system](paths['one_step'])[1]
    if system == 'l96':
        mean, std = l96_normaliser(paths['one_step'], meta, norm_from)
        raw = simulate_l96(seed=test_seed, steps=steps, burnin=2000)
        s_mean, s_std = float(np.mean(mean)), float(np.mean(std))   # original metric used scalars
    else:
        mean, std = float(meta['normalizer_mean']), float(meta['normalizer_std'])
        raw = simulate_ks(seed=test_seed, steps=steps, burnin=2000, mean=KS_TRAIN_MEAN)
        s_mean, s_std = mean, std
    test = (raw - mean) / std
    starts = np.arange(0, len(test) - max(HORIZONS) - 1, spacing)
    rows = []
    for arm, m in models.items():
        errs = batched_errors(m, test, starts, HORIZONS)
        for h, (err, true, pred) in errs.items():
            num = np.sum((err * s_std) ** 2, axis=1)
            den = np.sum((true * s_std + s_mean) ** 2, axis=1)
            amp = np.linalg.norm(pred, axis=1) / np.linalg.norm(true, axis=1)
            for w, (a, b, e, r) in enumerate(zip(num, den, np.linalg.norm(err, axis=1), amp)):
                rows.append(dict(seed=seed, arm=arm, window=w, start=int(starts[w]), h=h,
                                 rel2=float(a / b), err_norm=float(e), amp_ratio=float(r)))
    return pd.DataFrame(rows)


def paired_summary(df, n_boot=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    out = []
    df = df[df.arm.isin(['one_step', 'rollout'])]
    for (seed, h), g in df.groupby(['seed', 'h']):
        a = g[g.arm == 'one_step'].sort_values('window').rel2.to_numpy()
        b = g[g.arm == 'rollout'].sort_values('window').rel2.to_numpy()
        change = 100 * (np.sqrt(b.mean() / a.mean()) - 1)
        idx = rng.integers(0, len(a), (n_boot, len(a)))
        boot = 100 * (np.sqrt(b[idx].mean(1) / a[idx].mean(1)) - 1)
        ga = g[g.arm == 'one_step'].sort_values('window').err_norm.to_numpy()
        gb = g[g.arm == 'rollout'].sort_values('window').err_norm.to_numpy()
        amp_a = g[g.arm == 'one_step'].amp_ratio.median(); amp_b = g[g.arm == 'rollout'].amp_ratio.median()
        out.append(dict(seed=seed, h=h, n_windows=len(a), median_amp_one=amp_a, median_amp_roll=amp_b,
                        rel_rmse_one=np.sqrt(a.mean()),
                        rel_rmse_roll=np.sqrt(b.mean()), paired_change_pct=change,
                        ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                        geo_mean_ratio=float(np.exp(np.mean(np.log(gb / ga)))),
                        frac_windows_improved=float(np.mean(gb < ga))))
    return pd.DataFrame(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["l96", "ks"], required=True)
    p.add_argument("--root", required=True, help="folder with seed_*/best_{one_step,rollout}.(pt|npz)")
    p.add_argument("--label", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--test-seed", type=int, default=20261019)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--spacing", type=int, default=None)
    p.add_argument("--norm-from", default=None)
    p.add_argument("--ext", default="pt")
    p.add_argument("--initial", default=None, help="optional pattern of the common initial checkpoint")
    a = p.parse_args()
    steps = a.steps or (40000 if a.system == 'l96' else 40000)
    spacing = a.spacing or 128
    frames = []
    for s in [int(v) for v in a.seeds.split(",")]:
        paths = {arm: f"{a.root}/seed_{s}/best_{arm}.{a.ext}" for arm in ('one_step', 'rollout')}
        if a.initial:
            paths['initial'] = a.initial.format(seed=s)
        nf = a.norm_from.format(seed=s) if a.norm_from else None
        frames.append(evaluate(a.system, paths, s, a.test_seed, steps, spacing, nf))
        print(a.label, 'seed', s, 'done', flush=True)
    df = pd.concat(frames)
    out = Path("results/current/intervention_eval") / a.label
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "window_errors.csv.gz", index=False, float_format="%.8g")
    summ = paired_summary(df)
    summ.to_csv(out / "paired_summary.csv", index=False)
    med = summ.groupby('h').agg(median_change=('paired_change_pct', 'median'),
                                seeds_improved=('paired_change_pct', lambda x: int((x < 0).sum())),
                                median_rel_one=('rel_rmse_one', 'median'), median_rel_roll=('rel_rmse_roll', 'median'))
    med.to_csv(out / "median_summary.csv")
    print(med.round(4).to_string())
