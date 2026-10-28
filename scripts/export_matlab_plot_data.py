"""Save the plotted values of every figure as CSV tables for the MATLAB figures.

python scripts/export_matlab_plot_data.py  ->  data/processed/figure_data/fig_*.csv

scripts/generate_all_figures.m draws the figures of the manuscript in MATLAB from these
tables. Every table has one column per plotted quantity and one row per surrogate family
and horizon (Figure 8: one row per bin of E_T/E_sat; Figure 10: three tables). The values
are computed from the saved results in results/current/ as in scripts/plot_figures.py, so
the MATLAB figures show the same numbers as the PDFs in figures/. Empty cells are values
that are not defined, for example the variance shares near saturation.
"""
from __future__ import annotations
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning

from analysis import load_windows, load_terms, covariance_shares, paired_attribution, system_of, PARTS5, LYAP, DT

warnings.simplefilter('ignore', ConstantInputWarning)   # e.g. the forced gain at T = 1: the cell stays empty

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed/figure_data'

FAMILY = {'l96_e20': 'L96-20', 'l96_e150': 'L96-150', 'l96_e2000': 'L96-2000', 'ks_pre': 'KS'}
# Figure 10: (case, evaluation, one-step campaign, rollout campaign)
CASES = [('L96-20', 'l96_e20_retry', 'l96_e20_ft_one', 'l96_e20_ft_roll'),
         ('L96-2000', 'l96_e2000_ct', 'l96_e2000_ct_one', 'l96_e2000_ct_roll'),
         ('KS', 'ks_retry', 'ks_one_step', 'ks_rollout')]


def gmean(x):
    x = np.asarray(x, float)
    return float(np.exp(np.mean(np.log(x[x > 0]))))


def within_seed_rho(g, col):
    """Median over seeds of the within-seed Spearman correlation between a window diagnostic and E_T."""
    return np.median([spearmanr(h[col], h.E).statistic for _, h in g.groupby('seed')])


def growth(labels):
    """Figure 2: median and 10-90% range over windows of E_T/E_sat and R_lin."""
    rows = []
    for lab in labels:
        g = load_windows(lab).groupby('T')
        x = g.lyap_time.first()
        rows.append(pd.DataFrame({
            'family': FAMILY[lab], 'T': x.index, 'Lambda': x.values,
            'E_over_Esat_med': g.E_sat.median().values, 'E_over_Esat_q10': g.E_sat.quantile(.1).values,
            'E_over_Esat_q90': g.E_sat.quantile(.9).values,
            'Rlin_med': g.Rlin.median().values, 'Rlin_q10': g.Rlin.quantile(.1).values,
            'Rlin_q90': g.Rlin.quantile(.9).values}))
    return pd.concat(rows, ignore_index=True)


def scalar(labels):
    """Figure 3: within-seed Spearman correlation with E_T (median over seeds) of scalar descriptors."""
    columns = [('Gmax', 'rho_Gmax'), ('forced_gain', 'rho_forced_gain'), ('sigma1J', 'rho_sigma1J'),
               ('Se', 'rho_S'), ('B', 'rho_B'), ('Sq', 'rho_Q'), ('Lin', 'rho_L')]
    rows = []
    for lab in labels:
        for T, g in load_windows(lab).groupby('T'):
            r = {'family': FAMILY[lab], 'T': T, 'Lambda': g.lyap_time.iloc[0]}
            r.update({name: within_seed_rho(g, k) for k, name in columns})
            rows.append(r)
    return pd.DataFrame(rows)


def budget(labels):
    """Figure 4: geometric means of the five factors and of E_T, and within-seed variance shares s_j."""
    factors = [('Se', 'S'), ('gbar', 'gbar'), ('eta', 'eta'), ('C', 'C'), ('N', 'N')]
    rows = []
    for lab in labels:
        for T, h in load_windows(lab).groupby('T'):
            r = {'family': FAMILY[lab], 'T': T, 'Lambda': h.lyap_time.iloc[0]}
            for k, name in factors + [('E', 'E')]:
                r[name] = gmean(h[k])
            # shares only where the error is well below saturation
            sh = covariance_shares(h, PARTS5) if T > 1 and h.E_sat.median() <= 0.5 else {}
            for k, name in factors:
                r['s_' + name] = sh.get('l_' + k, np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def efficiency(labels):
    """Figure 5: medians of eta_T and of its two nulls, and median and 10-90% range of alpha_T."""
    rows = []
    for lab in labels:
        g = load_windows(lab).groupby('T')
        x = g.lyap_time.first()
        rows.append(pd.DataFrame({
            'family': FAMILY[lab], 'T': x.index, 'Lambda': x.values,
            'eta_med': g.eta.median().values, 'eta_Sigma_med': g.eta_covmc.median().values,
            'eta_iso_med': g.eta_isomc.median().values,
            'alpha_med': g.orient.median().values, 'alpha_q10': g.orient.quantile(.1).values,
            'alpha_q90': g.orient.quantile(.9).values}))
    return pd.concat(rows, ignore_index=True)


def spectrum(labels):
    """Figure 6: means over defect-tail pairs of m_90 and of the participation number, actual defects and nulls."""
    cols = ('T', 'm90', 'm90_isoMC', 'm90_covMC', 'npart', 'npart_isoMC', 'npart_covMC')
    rows = []
    for lab in labels:
        t = load_terms(lab, usecols=lambda col: col in cols).dropna(subset=['m90'])
        g = t.groupby('T')
        T = np.array(g.size().index)
        rows.append(pd.DataFrame({
            'family': FAMILY[lab], 'T': T, 'Lambda': T * LYAP[system_of(lab)] * DT[system_of(lab)],
            'm90': g.m90.mean().values, 'm90_Sigma': g.m90_covMC.mean().values, 'm90_iso': g.m90_isoMC.mean().values,
            'Npart': g.npart.mean().values, 'Npart_Sigma': g.npart_covMC.mean().values,
            'Npart_iso': g.npart_isoMC.mean().values}))
    return pd.concat(rows, ignore_index=True)


def coherence(labels):
    """Figure 7: medians of T_eff^(1/2), kappa_T (with 10-90% range) and kappa_T^iid."""
    rows = []
    for lab in labels:
        df = load_windows(lab)
        df['sqrtTeff'] = np.sqrt(df.Teff)
        g = df.groupby('T')
        x = g.lyap_time.first()
        rows.append(pd.DataFrame({
            'family': FAMILY[lab], 'T': x.index, 'Lambda': x.values,
            'sqrtTeff_med': g.sqrtTeff.median().values, 'kappa_med': g.kappa.median().values,
            'kappa_q10': g.kappa.quantile(.1).values, 'kappa_q90': g.kappa.quantile(.9).values,
            'kappa_iid_med': g.kappa_iid.median().values}))
    return pd.concat(rows, ignore_index=True)


def validity(labels):
    """Figure 8: median |log R_lin| for windows binned by E_T/E_sat (bins with at least 20 windows)."""
    bins = np.logspace(-3, 1, 25)
    rows = []
    for lab in labels:
        df = load_windows(lab)
        df = df[df['T'] > 1]
        idx = np.digitize(df.E_sat, bins)
        for b in np.unique(idx):
            h = df[idx == b]
            if len(h) >= 20:
                rows.append({'family': FAMILY[lab], 'E_over_Esat': np.exp(np.mean(np.log(h.E_sat))),
                             'absLogRlin_med': np.median(np.abs(np.log(h.Rlin))), 'n_windows': len(h)})
    return pd.DataFrame(rows)


def ensemble(labels):
    """Figure 9: within-seed Spearman correlation with E_T of the reference-free scores, and median score/E_T."""
    rows = []
    for lab in labels:
        S = pd.read_csv(f"results/current/ensemble/{lab}_summary.csv")
        sysn = system_of(lab)
        d = pd.DataFrame({'family': FAMILY[lab], 'T': S['T'], 'Lambda': S['T'] * LYAP[sysn] * DT[sysn],
                          'rho_D': S.rho_D, 'rho_Lhat': S.rho_L_hat, 'rho_Shat': S.rho_S_hat,
                          'rho_e0hat': S.rho_e0_hat, 'rho_e0': S.rho_e0_true,
                          'rho_L': np.nan, 'rho_Swhite': np.nan, 'rho_Gmax_mdl': np.nan,
                          'ratio_D': S.ratio_D_E, 'ratio_Lhat': S.ratio_L_hat_E})
        if Path(f"results/current/campaign/{lab}").exists():      # retrospective L_T (true defects), T > 1
            rho = {T: within_seed_rho(h, 'Lin') for T, h in load_windows(lab).groupby('T') if T > 1}
            d['rho_L'] = d['T'].map(rho)
        st = Path(f"results/current/prospective/{lab}_summary.csv")
        if st.exists():                                             # scores built from the defect statistics
            P = pd.read_csv(st).set_index('T')
            d['rho_Swhite'] = d['T'].map(P.rho_S_white)
            d['rho_Gmax_mdl'] = d['T'].map(P.rho_Gmax_mdl)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def intervention(cases):
    """Figure 10: paired change per seed, change of the log-factors and amplitude ratios (three tables).

    The column "family" holds the case of the fine-tuning study (L96-20, L96-2000 or KS).
    """
    factors = [('dl_Se', 'dS'), ('dl_gbar', 'dgbar'), ('dl_eta', 'deta'), ('dl_C', 'dC'), ('dl_N', 'dN')]
    change, attribution, amplitude = [], [], []
    for case, ev, cref, cnew in cases:
        P = pd.read_csv(f"results/current/intervention_eval/{ev}/paired_summary.csv")
        med = P.groupby('h').paired_change_pct.median()
        change.append(pd.DataFrame({'family': case, 'seed': P.seed, 'T': P.h, 'change': P.paired_change_pct,
                                    'ci_lo': P.ci_lo, 'ci_hi': P.ci_hi, 'change_median': P.h.map(med)}))
        if Path(f"results/current/campaign/{cref}").exists() and Path(f"results/current/campaign/{cnew}").exists():
            per_seed, medf = paired_attribution(cref, cnew)
            Ts = [t for t in medf.index if t > 2]
            f = pd.DataFrame({'family': case, 'T': Ts})
            for k, name in factors + [('dlogE', 'dE')]:          # median over seeds of the mean change, in %
                f[name] = 100 * medf.loc[Ts, k].values
            attribution.append(f)
        Wd = pd.read_csv(f"results/current/intervention_eval/{ev}/window_errors.csv.gz")
        for arm in ['initial', 'one_step', 'rollout']:
            m = Wd[Wd.arm == arm].groupby('h').amp_ratio.median()
            if len(m):
                amplitude.append(pd.DataFrame({'family': case, 'arm': arm, 'T': m.index, 'amp_ratio': m.values}))
    return (pd.concat(change, ignore_index=True), pd.concat(attribution, ignore_index=True),
            pd.concat(amplitude, ignore_index=True))


def main():
    os.chdir(ROOT)                    # the saved results are read relative to the repository root
    OUT.mkdir(parents=True, exist_ok=True)
    change, attribution, amplitude = intervention(CASES)
    tables = {
        'fig_growth': growth(['l96_e20', 'l96_e150', 'l96_e2000', 'ks_pre']),
        'fig_scalar': scalar(['l96_e2000', 'ks_pre']),
        'fig_budget': budget(['l96_e2000', 'ks_pre', 'l96_e20']),
        'fig_efficiency': efficiency(['l96_e2000', 'ks_pre']),
        'fig_spectrum': spectrum(['l96_e2000', 'ks_pre']),
        'fig_coherence': coherence(['l96_e2000', 'ks_pre']),
        'fig_validity': validity(['l96_e20', 'l96_e150', 'l96_e2000', 'ks_pre']),
        'fig_ensemble': ensemble(['l96_e2000', 'l96_e20', 'ks_pre']),
        'fig_intervention_change': change,
        'fig_intervention_factors': attribution,
        'fig_intervention_amplitude': amplitude,
    }
    for name, table in tables.items():
        table.to_csv(OUT / f'{name}.csv', index=False)
        print(f'{name}.csv: {len(table)} rows, columns {", ".join(table.columns)}')


if __name__ == '__main__':
    main()
