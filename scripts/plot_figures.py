"""Figures of the manuscript (Matplotlib, vector PDF).

python scripts/plot_figures.py [growth scalar budget ...]  ->  figures/fig_*.pdf (and PNG previews)
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from analysis import load_windows, load_terms, covariance_shares, PARTS5, PARTS7, LYAP, DT, paired_attribution

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         green="#008300", violet="#6250d6", red="#e34948", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df")
SERIES = [C['blue'], C['orange'], C['aqua'], C['yellow'], C['magenta'], C['green'], C['violet'], C['red']]
MARK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5, "legend.fontsize": 9.0,
    "xtick.labelsize": 9.0, "ytick.labelsize": 9.0, "axes.edgecolor": C['ink2'], "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": C['grid'], "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.3, "lines.markersize": 3.6, "xtick.color": C['ink2'], "ytick.color": C['ink2'],
    "axes.labelcolor": C['ink'], "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
W = 6.0   # JMLR text width (in)

TITLES = {'l96_e2000': 'L96-2000', 'l96_e150': 'L96-150',
          'l96_e20': 'L96-20', 'ks_pre': 'KS (FNO)', 'ks_one_step': 'KS (FNO, one-step arm)'}


def gmean(x):
    x = np.asarray(x, float)
    return float(np.exp(np.mean(np.log(x[x > 0]))))


LYAP_LABEL = r'$\Lambda=\lambda_1T\Delta t$ (Lyapunov times)'


def lyap_axis(ax, label=True):
    ax.set_xscale('log')
    if label:
        ax.set_xlabel(LYAP_LABEL)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", metadata={'CreationDate': None})   # no time stamp: identical bytes on every run
    fig.savefig(OUT / f"{name}.png", dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_growth(labels):
    """Median error relative to saturation versus horizon, and R_lin quantiles."""
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.3))
    for i, lab in enumerate(labels):
        df = load_windows(lab)
        g = df.groupby('T')
        x = g.lyap_time.first()
        axs[0].plot(x, g.E_sat.median(), marker=MARK[i], color=SERIES[i], label=TITLES[lab])
        axs[0].fill_between(x, g.E_sat.quantile(.1), g.E_sat.quantile(.9), color=SERIES[i], alpha=.12, lw=0)
        axs[1].plot(x, g.Rlin.median(), marker=MARK[i], color=SERIES[i])
        axs[1].fill_between(x, g.Rlin.quantile(.1), g.Rlin.quantile(.9), color=SERIES[i], alpha=.12, lw=0)
    axs[0].set_yscale('log'); axs[0].set_ylabel(r'$E_T/E_{\rm sat}$')
    axs[0].axhline(1, color=C['ink2'], lw=.7)
    axs[1].set_yscale('log'); axs[1].set_ylabel(r'$R_{\rm lin}=L_T/E_T$'); axs[1].axhline(1, color=C['ink2'], lw=.7)
    for ax in axs:
        lyap_axis(ax)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(0.5, 1.04), ncol=4, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.83])
    save(fig, 'fig_growth')


def fig_scalar(labels):
    keys = [('Gmax', r'$G_{\max}$', 'ink2', '--'), ('forced_gain', 'forced gain', 'violet', '--'),
            ('sigma1J', r'$\sigma_1(J_0)$', 'yellow', '--'),
            ('Se', r'$S_T$', 'blue', '-'), ('B', r'$B_T$', 'orange', '-'), ('Sq', r'$Q_T$', 'aqua', '-'),
            ('Lin', r'$L_T=\|\delta^{\rm lin}_T\|$', 'magenta', '-')]
    fig, axs = plt.subplots(1, len(labels), figsize=(W, 2.4), sharey=True)
    from scipy.stats import spearmanr
    for ax, lab in zip(np.atleast_1d(axs), labels):
        df = load_windows(lab)
        for j, (k, name, col, ls) in enumerate(keys):
            rows = []
            for T, g in df.groupby('T'):
                if T == 1 and k in ('Gmax',):
                    pass
                rows.append((g.lyap_time.iloc[0], np.median([spearmanr(h[k], h.E).statistic for _, h in g.groupby('seed')])))
            r = np.array(rows)
            ax.plot(r[:, 0], r[:, 1], ls=ls, marker=MARK[j % 8], color=C[col], label=name)
        ax.set_title(TITLES[lab]); lyap_axis(ax); ax.axhline(0, color=C['ink2'], lw=.6)
    np.atleast_1d(axs)[0].set_ylabel('within-seed Spearman with $E_T$')
    np.atleast_1d(axs)[-1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    save(fig, 'fig_scalar')


def fig_budget(labels):
    names = [('Se', r'magnitude $S_T$'), ('gbar', r'worst-case gain $\bar g_T$'), ('eta', r'efficiency $\eta_T$'),
             ('C', r'cancellation $C_T$'), ('N', r'nonlinear $N_T$')]
    fig, axs = plt.subplots(2, len(labels), figsize=(W, 4.0), sharex='col', layout='constrained')
    for c, lab in enumerate(labels):
        df = load_windows(lab)
        g = df.groupby('T')
        x = g.lyap_time.first().to_numpy()
        ax = axs[0, c]
        for j, (k, nm) in enumerate(names):
            ax.plot(x, g[k].apply(gmean), marker=MARK[j], color=SERIES[j], label=nm)
        ax.plot(x, g.E.apply(gmean), color=C['ink'], lw=1.8, label=r'error $E_T$')
        ax.set_yscale('log'); ax.set_title(TITLES[lab])
        ax = axs[1, c]
        sh = []
        for T, h in g:
            if T == 1 or h.E_sat.median() > 0.5:     # shares only where the error is well below saturation
                continue
            sh.append((h.lyap_time.iloc[0], covariance_shares(h, PARTS5)))
        for j, (k, nm) in enumerate(names):
            ax.plot([s[0] for s in sh], [s[1]['l_' + k] for s in sh], marker=MARK[j], color=SERIES[j])
        ax.axhline(0, color=C['ink2'], lw=.6); ax.set_ylim(-0.25, 1.05)
        lyap_axis(ax, label=False)
    axs[0, 0].set_ylabel('geometric mean'); axs[1, 0].set_ylabel(r'variance share $s_j$')
    fig.supxlabel(LYAP_LABEL, fontsize=9.5)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside upper center', ncol=3)
    save(fig, 'fig_budget')


def fig_efficiency(labels):
    fig, axs = plt.subplots(2, len(labels), figsize=(W, 3.6), sharex='col')
    for c, lab in enumerate(labels):
        df = load_windows(lab)
        df['alpha'] = df.orient                     # unbiased Monte Carlo norm null
        g = df.groupby('T'); x = g.lyap_time.first()
        ax = axs[0, c]
        ax.plot(x, g.eta.median(), marker='o', color=C['blue'], label=r'actual $\eta_T$')
        ax.plot(x, g.eta_covmc.median(), marker='s', color=C['orange'], ls='--', label=r'covariance null $\eta^\Sigma_T$')
        ax.plot(x, g.eta_isomc.median(), marker='^', color=C['aqua'], ls=':', label=r'isotropic null $\eta^{\rm iso}_T$')
        d = df.groupby('T').apply(lambda h: h.attrs) if False else None
        ax.set_ylim(0, 1.03); ax.set_title(TITLES[lab])
        ax = axs[1, c]
        ax.plot(x, g.alpha.median(), marker='o', color=C['violet'], label=r'median $\alpha_T=\eta_T/\eta^\Sigma_T$')
        ax.fill_between(x, g.alpha.quantile(.1), g.alpha.quantile(.9), color=C['violet'], alpha=.15, lw=0, label='10–90%')
        ax.axhline(1, color=C['ink2'], lw=.7)
        lyap_axis(ax)
    axs[0, 0].set_ylabel('efficiency'); axs[1, 0].set_ylabel(r'orientation ratio $\alpha_T$')
    axs[0, -1].legend(loc='upper right'); axs[1, -1].legend(loc='lower left')
    fig.tight_layout()
    save(fig, 'fig_efficiency')


def fig_coherence(labels):
    fig, axs = plt.subplots(1, len(labels), figsize=(W, 2.3), sharey=True, layout='constrained')
    for ax, lab in zip(np.atleast_1d(axs), labels):
        df = load_windows(lab)
        df['sqrtTeff'] = np.sqrt(df.Teff)
        g = df.groupby('T'); x = g.lyap_time.first()
        ax.plot(x, g.sqrtTeff.median(), marker='s', color=C['ink2'], ls='--', label=r'coherent limit $T_{\rm eff}^{1/2}$')
        ax.plot(x, g.kappa.median(), marker='o', color=C['blue'], label=r'coherence $\kappa_T$')
        ax.fill_between(x, g.kappa.quantile(.1), g.kappa.quantile(.9), color=C['blue'], alpha=.15, lw=0)
        ax.plot(x, g.kappa_iid.median(), marker='^', color=C['orange'], ls=':', label=r'independent defects $\kappa^{\rm iid}_T$')
        ax.set_yscale('log'); ax.set_title(TITLES[lab]); lyap_axis(ax, label=False)
    np.atleast_1d(axs)[0].set_ylabel('coherence')
    fig.supxlabel(LYAP_LABEL, fontsize=9.5)
    fig.legend(*np.atleast_1d(axs)[0].get_legend_handles_labels(), loc='outside right center')
    save(fig, 'fig_coherence')


def fig_spectrum(labels):
    cols = ('T', 'm90', 'm90_isoMC', 'm90_covMC', 'npart', 'npart_isoMC', 'npart_covMC')
    fig, axs = plt.subplots(2, len(labels), figsize=(W, 3.6), sharex='col', squeeze=False)
    for c, lab in enumerate(labels):
        t = load_terms(lab, usecols=lambda col: col in cols).dropna(subset=['m90'])
        sysn = 'l96' if lab.startswith('l96') else 'ks'
        g = t.groupby('T'); x = np.array(g.size().index) * LYAP[sysn] * DT[sysn]
        for r, (q, lab_q) in enumerate([('m90', r'$m_{90}$'), ('npart', r'$N_{\rm part}$')]):
            ax = axs[r, c]
            ax.plot(x, g[q].mean(), marker='o', color=C['blue'], label='actual defects')
            ax.plot(x, g[q + '_covMC'].mean(), marker='s', color=C['orange'], ls='--', label='covariance null')
            ax.plot(x, g[q + '_isoMC'].mean(), marker='^', color=C['aqua'], ls=':', label='isotropic null')
            ax.set_yscale('log'); ax.set_ylabel(f'mean {lab_q}' if c == 0 else '')
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            if r == 0:
                ax.set_title(TITLES[lab])
            else:
                lyap_axis(ax)
    axs[0, 0].legend(loc='lower left')
    fig.tight_layout()
    save(fig, 'fig_spectrum')


def fig_validity(labels):
    fig, ax = plt.subplots(1, 1, figsize=(W * 0.55, 2.4))
    bins = np.logspace(-3, 1, 25)
    for i, lab in enumerate(labels):
        df = load_windows(lab)
        df = df[df['T'] > 1]
        idx = np.digitize(df.E_sat, bins)
        rows = []
        for b in np.unique(idx):
            h = df[idx == b]
            if len(h) >= 20:
                rows.append((np.exp(np.mean(np.log(h.E_sat))), np.median(np.abs(np.log(h.Rlin)))))
        r = np.array(rows)
        ax.plot(r[:, 0], r[:, 1], marker=MARK[i], color=SERIES[i], label=TITLES[lab])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$E_T/E_{\rm sat}$'); ax.set_ylabel(r'median $|\log R_{\rm lin}|$')
    ax.legend(loc='upper left', ncol=1, fontsize=9.0, handlelength=1.6, borderaxespad=0.3)
    fig.tight_layout()
    save(fig, 'fig_validity')


def fig_ensemble(labels):
    """Reference-free scores: within-seed Spearman correlation with E_T and calibration of the spread."""
    keys = [('D', r'ensemble spread $D_T$', 'blue', '-', 'o'), ('L_hat', r'propagated proxy $\hat L_T$', 'magenta', '-', 's'),
            ('S_hat', r'proxy magnitude $\hat S_T$', 'aqua', '-', '^'), ('e0_hat', r'one-step proxy $\|\hat{e}_0\|$', 'yellow', '-', 'D'),
            ('e0_true', r'true one-step defect $\|e_0\|$', 'ink2', '--', 'v')]
    n = len(labels)
    fig, axs = plt.subplots(2, n, figsize=(W, 4.6), sharex='col', squeeze=False, layout='constrained')
    for c, lab in enumerate(labels):
        S = pd.read_csv(f"results/current/ensemble/{lab}_summary.csv")
        sysn = 'l96' if lab.startswith('l96') else 'ks'
        x = S['T'] * LYAP[sysn] * DT[sysn]
        ax = axs[0, c]
        for k, nm, col, ls, mk in keys:
            ax.plot(x, S['rho_' + k], ls=ls, marker=mk, color=C[col], label=nm)
        if Path(f"results/current/campaign/{lab}").exists():
            from scipy.stats import spearmanr
            Wd = load_windows(lab)
            rr = [(h.lyap_time.iloc[0], np.median([spearmanr(q.Lin, q.E).statistic for _, q in h.groupby('seed')]))
                  for _, h in Wd.groupby('T') if h['T'].iloc[0] > 1]
            ax.plot([r[0] for r in rr], [r[1] for r in rr], ls='--', color=C['green'], lw=1.0,
                    label=r'retrospective $L_T$ (true defects)')
        st = Path(f"results/current/prospective/{lab}_summary.csv")
        if st.exists():
            P = pd.read_csv(st)
            xs = P['T'] * LYAP[sysn] * DT[sysn]
            ax.plot(xs, P['rho_S_white'], ls=':', marker='x', color=C['red'], label=r'statistical score $S^{\rm white}_T$')
            ax.plot(xs, P['rho_Gmax_mdl'], ls=':', marker='+', color=C['violet'], label=r'$G_{\max}$ (model path)')
        ax.axhline(0, color=C['ink2'], lw=.6); ax.set_ylim(-0.4, 1.02); ax.set_title(TITLES[lab])
        ax = axs[1, c]
        ax.plot(x, S['ratio_D_E'], marker='o', color=C['blue'], label=r'$D_T/E_T$')
        ax.plot(x, S['ratio_L_hat_E'], marker='s', color=C['magenta'], label=r'$\hat L_T/E_T$')
        ax.axhline(1, color=C['ink2'], lw=.7); ax.set_ylim(0.5, 2.0)
        ax.set_xscale('log')
    axs[0, 0].set_ylabel('within-seed Spearman\nwith $E_T$'); axs[1, 0].set_ylabel('median score /\nrealized error')
    hs, ls_ = [], []
    for ax in axs[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls_:
                hs.append(h); ls_.append(l)
    fig.legend(hs, ls_, loc='outside upper center', ncol=2, fontsize=9.0)
    axs[1, 0].legend(loc='upper left', fontsize=9.0, handlelength=1.6)
    fig.supxlabel(LYAP_LABEL, fontsize=9.5)
    save(fig, 'fig_ensemble')


def fig_intervention(cases):
    """cases: list of (title, eval_label, campaign_ref, campaign_new)."""
    n = len(cases)
    fig, axs = plt.subplots(3, n, figsize=(W + 0.6, 6.2), squeeze=False)
    names = [('dl_Se', r'$S_T$'), ('dl_gbar', r'$\bar g_T$'), ('dl_eta', r'$\eta_T$'), ('dl_C', r'$C_T$'), ('dl_N', r'$N_T$')]
    for c, (title, ev, cref, cnew) in enumerate(cases):
        ax = axs[0, c]
        P = pd.read_csv(f"results/current/intervention_eval/{ev}/paired_summary.csv")
        for s, g in P.groupby('seed'):
            off = 1 + 0.04 * (s - 2)
            ax.errorbar(g.h * off, g.paired_change_pct, yerr=[g.paired_change_pct - g.ci_lo, g.ci_hi - g.paired_change_pct],
                        fmt=MARK[s], color=SERIES[s], ms=3, lw=0.8, capsize=0, label=f'seed {s}')
        med = P.groupby('h').paired_change_pct.median()
        ax.plot(med.index, med.values, color=C['ink'], lw=1.4, label='median')
        ax.axhline(0, color=C['ink2'], lw=.6); ax.set_xscale('log', base=2); ax.set_title(title, fontsize=9.5)
        ax.set_xlabel('horizon $T$ (steps)')
        if c == 0:
            ax.set_ylabel('paired change in\nrel. RMSE (%)')
        # attribution
        ax = axs[1, c]
        if cref and Path(f"results/current/campaign/{cref}").exists() and Path(f"results/current/campaign/{cnew}").exists():
            per_seed, medf = paired_attribution(cref, cnew)
            Ts = [t for t in medf.index if t > 2]
            x = np.arange(len(Ts)); wbar = 0.14
            for j, (k, nm) in enumerate(names):
                ax.bar(x + (j - 2) * wbar, 100 * medf.loc[Ts, k], width=wbar * 0.92, color=SERIES[j], label=nm)
            ax.plot(x, 100 * medf.loc[Ts, 'dlogE'], 'k_', ms=9, mew=1.6, label=r'$E_T$')
            ax.set_xticks(x); ax.set_xticklabels([str(t) for t in Ts], fontsize=9, rotation=90)
            ax.axhline(0, color=C['ink2'], lw=.6); ax.set_xlabel('horizon $T$ (steps)')
        if c == 0:
            ax.set_ylabel('median change of\nlog-factor (%)')
        # amplitude
        ax = axs[2, c]
        Wd = pd.read_csv(f"results/current/intervention_eval/{ev}/window_errors.csv.gz")
        for j, arm in enumerate(['initial', 'one_step', 'rollout']):
            g = Wd[Wd.arm == arm]
            if len(g):
                m = g.groupby('h').amp_ratio.median()
                ax.plot(m.index, m.values, marker=MARK[j], color=[C['ink2'], C['blue'], C['orange']][j],
                        ls=[':', '--', '-'][j], label={'initial': 'initial checkpoint', 'one_step': 'one-step arm', 'rollout': 'rollout arm'}[arm])
        ax.axhline(1, color=C['ink2'], lw=.6); ax.set_xscale('log', base=2); ax.set_xlabel('horizon $T$ (steps)')
        if c == 0:
            ax.set_ylabel('median forecast/reference\namplitude')
    axs[0, -1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9.0)
    axs[1, -1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9.0)
    axs[2, -1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9.0)
    fig.tight_layout()
    save(fig, 'fig_intervention')


if __name__ == "__main__":
    import sys
    avail = sorted(Path(p).name for p in glob.glob("results/current/campaign/*"))
    main = [l for l in ['l96_e2000', 'ks_pre', 'l96_e20'] if l in avail]
    allm = [l for l in ['l96_e20', 'l96_e150', 'l96_e2000', 'ks_pre'] if l in avail]
    two = [l for l in ['l96_e2000', 'ks_pre'] if l in avail]
    which = sys.argv[1:] or ['growth', 'scalar', 'budget', 'efficiency', 'coherence', 'spectrum', 'validity',
                             'ensemble', 'intervention']
    for w in which:
        if w == 'ensemble':
            fig_ensemble([l for l in ['l96_e2000', 'l96_e20', 'ks_pre'] if Path(f"results/current/ensemble/{l}_summary.csv").exists()])
        elif w == 'intervention':
            fig_intervention([('L96-20 (fine-tuned)', 'l96_e20_retry', 'l96_e20_ft_one', 'l96_e20_ft_roll'),
                              ('L96-2000 (continued)', 'l96_e2000_ct', 'l96_e2000_ct_one', 'l96_e2000_ct_roll'),
                              ('KS (fine-tuned)', 'ks_retry', 'ks_one_step', 'ks_rollout')])
        else:
            globals()['fig_' + w](allm if w in ('growth', 'validity') else (main if w == 'budget' else two))
        print('wrote', w)
