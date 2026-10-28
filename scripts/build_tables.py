"""LaTeX tables and the numbers quoted in the manuscript.

python scripts/build_tables.py   ->  tables/tab_*.tex, tables/horizon_summary_*.csv, tables/paper_numbers.json
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis import (load_windows, load_terms, covariance_shares, cluster_bootstrap_shares, PARTS5, PARTS7,
                         paired_attribution, LYAP, DT)

OUT = Path("tables"); OUT.mkdir(exist_ok=True)
FAMILIES = [('l96_e20', 'L96-20'), ('l96_e150', 'L96-150'), ('l96_e2000', 'L96-2000'), ('ks_pre', 'KS')]
NUM = {}


def within_seed_rho(g, col, target='E'):
    return float(np.median([spearmanr(h[col], h[target]).statistic for _, h in g.groupby('seed')]))


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '--'
    return f"{x:.{nd}f}"


def horizon_rows(lab):
    df = load_windows(lab)
    if 'orient' in df:                       # unbiased Monte Carlo norm null
        df['alpha'] = df.orient
        df['eta_null'] = df.eta_covmc
    else:                                    # legacy outputs (RMS null)
        df['alpha'] = df.eta / df.eta_cov
        df['eta_null'] = df.eta_cov
    if 'kappa_iid' not in df:
        df['kappa_iid'] = np.nan
    rows = []
    for T, g in df.groupby('T'):
        r = dict(T=int(T), Lam=float(g.lyap_time.iloc[0]), Esat=g.E_sat.median(), Rlin=g.Rlin.median(),
                 Rlin10=g.Rlin.quantile(.1), Rlin90=g.Rlin.quantile(.9),
                 rhoG=within_seed_rho(g, 'Gmax'), rhoF=within_seed_rho(g, 'forced_gain'),
                 rhoJ=within_seed_rho(g, 'sigma1J'), rhoS=within_seed_rho(g, 'Se'), rhoB=within_seed_rho(g, 'B'),
                 rhoQ=within_seed_rho(g, 'Sq'), rhoL=within_seed_rho(g, 'Lin'),
                 gbar=g.gbar.median(), eta=g.eta.median(), eta_cov=g.eta_null.median(),
                 eta_iso=(g.eta_isomc if 'eta_isomc' in g else g.eta_iso).median(),
                 kappa_iid=g.kappa_iid.median(), bias_frac=(g.bias_frac.median() if 'bias_frac' in g else np.nan),
                 alpha=g.alpha.median(), alpha10=g.alpha.quantile(.1), alpha90=g.alpha.quantile(.9),
                 C=g.C.median(), kappa=g.kappa.median(), sqrtTeff=np.sqrt(g.Teff).median(),
                 BoverE=(g.B / g.E).median(), Gmax=g.Gmax.median())
        if T > 1:
            sh = covariance_shares(g, PARTS5)
            r.update({'s_' + k[2:]: v for k, v in sh.items()})
            sh7 = covariance_shares(g, PARTS7)
            r.update({'s7_' + k[2:]: v for k, v in sh7.items()})
        rows.append(r)
    return pd.DataFrame(rows)


def latex_horizon_table(lab, name, H):
    lines = [r"\begin{tabular}{rrrrrrrrrrrrrrr}", r"\toprule",
             r"$T$ & $\Lambda$ & $E_T/E_{\rm sat}$ & $\rho_G$ & $\rho_{\mathcal L}$ & $\rho_S$ & $\rho_L$ & $R_{\rm lin}$ & $\bar g_T$ & $\eta_T$ & $\eta^\Sigma_T$ & $\alpha_T$ & $C_T$ & $\kappa_T$ & $\kappa^{\rm iid}_T$\\",
             r"\midrule"]
    for _, r in H.iterrows():
        # fmt() writes '--' for undefined values, such as the forced-gain correlation at T=1 (the gain is one in every window)
        lines.append(f"{int(r['T'])} & {r.Lam:.2f} & {fmt(r.Esat, 3)} & {fmt(r.rhoG)} & {fmt(r.rhoF)} & {fmt(r.rhoS)} & {fmt(r.rhoL, 3)} & "
                     f"{fmt(r.Rlin, 3)} & {fmt(r.gbar)} & {fmt(r.eta, 3)} & {fmt(r.eta_cov, 3)} & {fmt(r.alpha, 3)} & {fmt(r.C, 3)} & {fmt(r.kappa)} & {fmt(r.kappa_iid)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"tab_horizons_{lab}.tex").write_text("\n".join(lines) + "\n")


def latex_share_table(fams):
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Surrogate & $T$ & $\Lambda$ & $s_S$ & $s_{\bar g}$ & $s_{\eta^\Sigma}$ & $s_\alpha$ & $s_{C}$ & $s_N$\\", r"\midrule"]
    for lab, name in fams:
        H = horizon_rows(lab)
        for _, r in H[(H['T'] > 1) & (H.Esat < 0.5)].iterrows():
            lines.append(f"{name} & {int(r['T'])} & {r.Lam:.2f} & {r.s_Se:.2f} & {r.s_gbar:.2f} & {r.s7_eta_cov:.2f} & "
                         f"{r.s7_orient:.2f} & {r.s_C:.2f} & {r.s_N:.2f}\\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (OUT / "tab_shares.tex").write_text("\n".join(lines) + "\n")


def fidelity_table():
    """Surrogate accuracy and free-run behavior (Table in Section 6.1)."""
    rows = []
    for lab, name in FAMILIES:
        f = Path(f"results/current/fidelity/{lab}.json")
        metas = [json.loads(p.read_text()) for p in sorted(Path(f"results/current/campaign/{lab}").glob("seed_*/meta.json"))]
        if not metas:
            continue
        mse = np.array([m['test_one_step_mse'] for m in metas])
        pers = np.array([m['persistence_mse'] for m in metas])
        r = dict(name=name, mse_med=float(np.median(mse)), mse_min=float(mse.min()), mse_max=float(mse.max()),
                 ratio_med=float(np.median(pers / mse)))
        if f.exists():
            F = json.loads(f.read_text())
            lr = np.array([v['lyap_reference_path'][0] for v in F.values()])
            lf = np.array([v['lyap_free_run'][0] for v in F.values()], dtype=float)
            blow = [v['blow_up_step'] for v in F.values()]
            amp = np.array([v['amp_free_mean'] for v in F.values()])
            ampr = np.array([v['amp_ref_mean'] for v in F.values()])
            r.update(lyap_ref=float(lr.mean()), lyap_ref_sd=float(lr.std(ddof=1)) if len(lr) > 1 else 0.0,
                     lyap_free=float(np.nanmean(lf)) if np.isfinite(lf).any() else float('nan'),
                     n_blow=int(sum(b is not None for b in blow)),
                     blow_steps=[b for b in blow if b is not None], amp_ratio=float(np.median(amp / ampr)))
        rows.append(r)
    return pd.DataFrame(rows)


def representative_table(cases):
    """cases: list of (label, name, T). Writes tables/tab_representative.tex."""
    cols = []
    for lab, name, T in cases:
        H = horizon_rows(lab)
        r = H[H['T'] == T].iloc[0]
        cols.append((name, r))
    rows = [
        (r'$\Lambda$', 'Lam', 2), (r'$E_T/E_{\rm sat}$', 'Esat', 3), (r'$R_{\rm lin}$', 'Rlin', 3),
        (r'$B_T/E_T$', 'BoverE', 1), (r'$\bar g_T$', 'gbar', 1), (r'$\eta_T$', 'eta', 2), (r'$\eta^\Sigma_T$', 'eta_cov', 2),
        (r'$\eta^{\rm iso}_T$', 'eta_iso', 2), (r'$\alpha_T$', 'alpha', 2), (r'$C_T$', 'C', 2), (r'$\kappa_T$', 'kappa', 1), (r'$\kappa^{\rm iid}_T$', 'kappa_iid', 2),
        (r'$T_{\rm eff}^{1/2}$', 'sqrtTeff', 1), (r'$s_S$', 's_Se', 2), (r'$s_{\bar g}$', 's_gbar', 2), (r'$s_\eta$', 's_eta', 2),
        (r'$s_C$', 's_C', 2), (r'$s_N$', 's_N', 2)]
    head = " & ".join([''] + [f"{n}, $T={int(r['T'])}$" for n, r in cols])
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", head + r"\\", r"\midrule"]
    for label, key, nd in rows:
        lines.append(" & ".join([label] + [f"{r[key]:.{nd}f}" for _, r in cols]) + r"\\")
        if key in ('Rlin', 'sqrtTeff'):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_representative.tex").write_text("\n".join(lines) + "\n")
    return cols


def intervention_tables(cases):
    """cases: list of (name, eval_label). Writes tab_intervention.tex (medians) and tab_intervention_seeds.tex."""
    hs = [16, 32, 64, 128]
    lines = [r"\begin{tabular}{l" + "r" * len(hs) + "}", r"\toprule",
             " & ".join(["Case"] + [f"$T={h}$" for h in hs]) + r"\\", r"\midrule"]
    seed_lines = [r"\begin{tabular}{ll" + "r" * len(hs) + "}", r"\toprule",
                  " & ".join(["Case", "Seed"] + [f"$T={h}$" for h in hs]) + r"\\", r"\midrule"]
    out = {}
    for name, ev in cases:
        P = pd.read_csv(f"results/current/intervention_eval/{ev}/paired_summary.csv")
        P = P[P.h.isin(hs)]
        med = P.groupby('h').paired_change_pct.median()
        n_imp = P.groupby('h').paired_change_pct.apply(lambda x: int((x < 0).sum()))
        lines.append(" & ".join([name] + [f"${med[h]:+.2f}$ ({n_imp[h]}/5)" for h in hs]) + r"\\")
        for s, g in P.groupby('seed'):
            g = g.set_index('h')
            seed_lines.append(" & ".join([name if s == 0 else '', str(s)] +
                                         [f"${g.loc[h, 'paired_change_pct']:+.2f}$ [{g.loc[h, 'ci_lo']:+.2f}, {g.loc[h, 'ci_hi']:+.2f}]" for h in hs]) + r"\\")
        seed_lines.append(r"\midrule")
        out[name] = dict(median=med.to_dict(), improved=n_imp.to_dict())
    lines += [r"\bottomrule", r"\end{tabular}"]
    seed_lines[-1] = r"\bottomrule"; seed_lines.append(r"\end{tabular}")
    (OUT / "tab_intervention.tex").write_text("\n".join(lines) + "\n")
    (OUT / "tab_intervention_seeds.tex").write_text("\n".join(seed_lines) + "\n")
    return out


def ensemble_table(cases):
    """cases: list of (label, name). Reference-free scores from ensemble disagreement."""
    out = {}
    for lab, name in cases:
        f = Path(f"results/current/ensemble/{lab}_summary.csv")
        if not f.exists():
            continue
        S = pd.read_csv(f)
        sysn = 'l96' if lab.startswith('l96') else 'ks'
        S['Lam'] = S['T'] * LYAP[sysn] * DT[sysn]
        pr = json.loads(Path(f"results/current/ensemble/{lab}_proxy.json").read_text())
        out[lab] = dict(summary=S.round(5).to_dict(orient='records'),
                        proxy_cos_median=float(np.median([q['median_cos'] for q in pr])),
                        proxy_norm_ratio_median=float(np.median([q['median_norm_ratio'] for q in pr])),
                        proxy_rho_norms_median=float(np.median([q['rho_norms'] for q in pr])))
        st = Path(f"results/current/prospective/{lab}_summary.csv")
        if st.exists():
            out[lab]['statistical'] = pd.read_csv(st).round(5).to_dict(orient='records')
    return out


def latex_ensemble_table(cases, horizons):
    """Within-seed Spearman correlation and AUROC of reference-free scores at selected horizons."""
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Surrogate & $T$ & $\Lambda$ & $\rho(\hat e_0)$ & $\rho(\hat S_T)$ & $\rho(\hat L_T)$ & $\rho(D_T)$ & AUROC$(D_T)$ & $D_T/E_T$\\",
             r"\midrule"]
    for lab, name in cases:
        f = Path(f"results/current/ensemble/{lab}_summary.csv")
        if not f.exists():
            continue
        S = pd.read_csv(f).set_index('T')
        sysn = 'l96' if lab.startswith('l96') else 'ks'
        for T in horizons[sysn]:
            r = S.loc[T]
            lines.append(f"{name} & {T} & {T * LYAP[sysn] * DT[sysn]:.2f} & {r.rho_e0_hat:.2f} & {r.rho_S_hat:.2f} & "
                         f"{r.rho_L_hat:.2f} & {r.rho_D:.2f} & {r.auroc_D:.2f} & {r.ratio_D_E:.2f}\\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (OUT / "tab_ensemble.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    avail = sorted(Path(p).name for p in glob.glob("results/current/campaign/*"))
    fams = [(l, n) for l, n in FAMILIES if l in avail]
    for lab, name in fams:
        H = horizon_rows(lab)
        H.to_csv(OUT / f"horizon_summary_{lab}.csv", index=False)
        latex_horizon_table(lab, name, H)
        NUM[lab] = H.round(5).to_dict(orient='records')
        print(name); print(H[['T', 'Lam', 'Esat', 'rhoG', 'rhoF', 'rhoS', 'rhoL', 'Rlin', 'gbar', 'eta', 'eta_cov', 'alpha',
                               'C', 'kappa', 'kappa_iid', 'sqrtTeff', 'BoverE']].round(3).to_string(index=False))
        print(H[['T', 's_Se', 's_gbar', 's_eta', 's_C', 's_N', 's7_eta_cov', 's7_orient', 's7_C_inc', 's7_kappa']].round(3).to_string(index=False))
    latex_share_table(fams)
    NUM['fidelity'] = fidelity_table().to_dict(orient='records')
    reps = [c for c in [('l96_e2000', 'L96', 40), ('l96_e2000', 'L96', 80), ('ks_pre', 'KS', 32), ('ks_pre', 'KS', 64),
                        ('ks_pre', 'KS', 128)] if c[0] in avail]
    representative_table(reps)
    # bootstrap intervals of the variance shares at the representative horizons
    NUM['share_ci'] = {}
    for lab, name, T in reps:
        g = load_windows(lab); g = g[g['T'] == T]
        lo, hi = cluster_bootstrap_shares(g, PARTS7)
        NUM['share_ci'][f"{lab}_T{T}"] = dict(lo=lo.round(3).to_dict(), hi=hi.round(3).to_dict(),
                                              point={k: round(v, 3) for k, v in covariance_shares(g, PARTS7).items()})
    iv = [(n, e) for n, e in [('L96-20', 'l96_e20_retry'), ('L96-2000', 'l96_e2000_ct'), ('KS', 'ks_retry')]
          if Path(f"results/current/intervention_eval/{e}/paired_summary.csv").exists()]
    NUM['intervention'] = intervention_tables(iv)
    NUM['intervention_attribution'] = {}
    for name, (a, b) in [('L96-20', ('l96_e20_ft_one', 'l96_e20_ft_roll')), ('L96-2000', ('l96_e2000_ct_one', 'l96_e2000_ct_roll')),
                         ('KS', ('ks_one_step', 'ks_rollout'))]:
        if a in avail and b in avail:
            per_seed, med = paired_attribution(a, b)
            NUM['intervention_attribution'][name] = dict(median=med.round(5).reset_index().to_dict(orient='records'),
                                                         per_seed=per_seed.round(5).to_dict(orient='records'))
    ens = [('l96_e2000', 'L96-2000'), ('l96_e150', 'L96-150'), ('l96_e20', 'L96-20'), ('ks_pre', 'KS')]
    NUM['ensemble'] = ensemble_table(ens)
    latex_ensemble_table([c for c in ens if c[0] != 'l96_e150'], {'l96': [10, 40, 80, 160], 'ks': [8, 32, 64, 128]})
    pgd = Path("results/current/pgd/l96_e2000/summary.csv")
    if pgd.exists():
        NUM['pgd'] = pd.read_csv(pgd).round(6).to_dict(orient='records')
    (OUT / "paper_numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
    print("wrote tables and paper_numbers.json")
