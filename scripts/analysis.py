"""Load and aggregate the diagnostic campaigns (also imported by build_tables.py and plot_figures.py).

python scripts/analysis.py l96_e20 l96_e150 l96_e2000 ks_pre

Reads results/current/campaign/<label>/seed_*/{windows.csv, terms.csv.gz, meta.json}
and writes the per-horizon summaries tables/horizons_<label>.csv and
tables/spectral_<label>.csv.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from realized_error.diagnostics import add_log_factors, covariance_shares

LYAP = {'l96': 1.565, 'ks': 0.0494}        # leading exponents of the reference systems (run_lyapunov.py)
DT = {'l96': 0.01, 'ks': 0.25}
OUT = Path("tables")
PARTS5 = ['l_Se', 'l_gbar', 'l_eta', 'l_C', 'l_N']
PARTS7 = ['l_Se', 'l_gbar', 'l_eta_cov', 'l_orient', 'l_C_inc', 'l_kappa', 'l_N']


def system_of(label):
    return 'l96' if label.startswith('l96') else 'ks'


def load_windows(label):
    frames = []
    for f in sorted(glob.glob(f"results/current/campaign/{label}/seed_*/windows.csv")):
        if not Path(f).with_name("meta.json").exists():      # seed still running
            continue
        df = pd.read_csv(f)
        meta = json.loads(Path(f).with_name("meta.json").read_text())
        df['E_sat'] = df['E'] / meta['saturation_scale']
        frames.append(df)
    df = add_log_factors(pd.concat(frames, ignore_index=True))
    sysn = system_of(label)
    df['lyap_time'] = LYAP[sysn] * DT[sysn] * df['T']
    return df


def load_terms(label, usecols=None):
    return pd.concat([pd.read_csv(f, usecols=usecols) for f in
                      sorted(glob.glob(f"results/current/campaign/{label}/seed_*/terms.csv.gz"))
                      if Path(f).with_name("meta.json").exists()], ignore_index=True)


def cluster_bootstrap_shares(g, parts, n_boot=400, rng=None):
    """Resample seeds, then initial states within each seed."""
    rng = rng or np.random.default_rng(1)
    seeds = g['seed'].unique()
    by_seed = {s: g[g.seed == s] for s in seeds}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, len(seeds), replace=True)
        parts_df = [by_seed[s].iloc[rng.integers(0, len(by_seed[s]), len(by_seed[s]))] for s in pick]
        draws.append(covariance_shares(pd.concat(parts_df), parts))
    D = pd.DataFrame(draws)
    return D.quantile(0.025), D.quantile(0.975)


def horizon_table(df):
    rows = []
    for T, g in df.groupby('T'):
        r = dict(T=T, lyap_time=g.lyap_time.iloc[0], n=len(g),
                 med_E=g.E.median(), med_E_sat=g.E_sat.median(),
                 med_Rlin=g.Rlin.median(), q10_Rlin=g.Rlin.quantile(.1), q90_Rlin=g.Rlin.quantile(.9),
                 med_gbar=g.gbar.median(), med_eta=g.eta.median(), med_eta_iso=g.eta_iso.median(),
                 med_eta_cov=g.eta_cov.median(), med_alpha_iso=g.alpha_iso.median(), med_alpha_cov=g.alpha_cov.median(),
                 med_C=g.C.median(), q10_C=g.C.quantile(.1), q90_C=g.C.quantile(.9),
                 med_C_inc=g.C_inc.median(), med_kappa=g.kappa.median(), med_Teff=g.Teff.median(),
                 med_Gmax=g.Gmax.median(), med_B_over_E=(g.B / g.E).median())
        for name, col in [('Gmax', 'Gmax'), ('rhoJ', 'rhoJ'), ('sigma1J', 'sigma1J'), ('froJ', 'froJ'),
                          ('Se', 'Se'), ('Sq', 'Sq'), ('Lin', 'Lin'), ('B', 'B'), ('forced', 'forced_gain'),
                          ('tail0', 'sigma1_tail0')]:
            r['rho_' + name] = spearmanr(g[col], g.E).statistic
        r['rho_Lin_within_seed'] = np.mean([spearmanr(h.Lin, h.E).statistic for _, h in g.groupby('seed')])
        r['rho_Gmax_within_seed'] = np.mean([spearmanr(h.Gmax, h.E).statistic for _, h in g.groupby('seed')])
        if T > 1:
            for p, v in covariance_shares(g, PARTS5).items():
                r['share5_' + p[2:]] = v
            for p, v in covariance_shares(g, PARTS7).items():
                r['share7_' + p[2:]] = v
        rows.append(r)
    return pd.DataFrame(rows)


def spectral_table(label):
    cols = ['seed', 'start', 'T', 'p1', 'P3', 'P5', 'm90', 'npart', 'alignment', 'efficiency']
    cols += [f'{q}_{n}' for q in ('p1', 'P3', 'P5', 'm90', 'npart') for n in ('iso', 'cov', 'isoMC', 'covMC')]
    cols += [f'{q}_{n}' for q in ('alignment', 'efficiency') for n in ('isoMC', 'covMC')]
    t = load_terms(label, usecols=lambda c: c in cols)
    t = t.dropna(subset=['p1'])
    return t.groupby('T').median(numeric_only=True).drop(columns=['seed', 'start']).reset_index()


if __name__ == "__main__":
    import sys
    OUT.mkdir(exist_ok=True)
    labels = sys.argv[1:] or sorted(Path(p).name for p in glob.glob("results/current/campaign/*"))
    for lab in labels:
        df = load_windows(lab)
        ht = horizon_table(df)
        ht.to_csv(OUT / f"horizons_{lab}.csv", index=False)
        st = spectral_table(lab)
        st.to_csv(OUT / f"spectral_{lab}.csv", index=False)
        print(lab)
        print(ht[['T', 'lyap_time', 'med_E_sat', 'med_Rlin', 'rho_Gmax', 'rho_Lin', 'med_eta', 'med_eta_cov',
                  'med_alpha_cov', 'med_C', 'med_kappa', 'med_B_over_E']].round(3).to_string(index=False))
        print(ht[['T'] + [c for c in ht.columns if c.startswith('share7')]].round(3).to_string(index=False))
        show = [c for c in ['T', 'm90', 'm90_isoMC', 'm90_covMC', 'npart', 'npart_isoMC', 'npart_covMC', 'p1', 'p1_isoMC',
                            'p1_covMC', 'alignment', 'alignment_isoMC', 'alignment_covMC', 'efficiency', 'efficiency_isoMC',
                            'efficiency_covMC'] if c in st]
        print(st[show].round(3).to_string(index=False))


# ------------------------------------------------------------------------
# paired attribution of an intervention (same windows, two model variants)
# ------------------------------------------------------------------------

def paired_attribution(label_ref, label_new, parts=PARTS5 + ['l_eta_cov', 'l_orient', 'l_C_inc', 'l_kappa']):
    """Mean over windows of log(new/ref) for E and for every exact factor, per seed and horizon.

    Because log E = sum of the five log-factors in each window, the mean
    log-change of E equals the sum of the mean log-changes of the factors.
    """
    a = load_windows(label_ref)
    b = load_windows(label_new)
    key = ['seed', 'start', 'T']
    m = a.merge(b, on=key, suffixes=('_ref', '_new'))
    rows = []
    for (seed, T), g in m.groupby(['seed', 'T']):
        r = dict(seed=seed, T=T, n=len(g), dlogE=float(np.mean(g['l_E_new'] - g['l_E_ref'])),
                 frac_windows_improved=float(np.mean(g['l_E_new'] < g['l_E_ref'])))
        for p in parts:
            r['d' + p] = float(np.mean(g[p + '_new'] - g[p + '_ref']))
        rows.append(r)
    per_seed = pd.DataFrame(rows)
    med = per_seed.groupby('T').median(numeric_only=True).drop(columns=['seed'])
    med['seeds_improved'] = per_seed.groupby('T')['dlogE'].apply(lambda x: int((x < 0).sum()))
    return per_seed, med
