"""Realised-error diagnostics, exact factorisation and null models.

For one reference window x_0..x_T and a learned one-step map f,

    e_k   = f(x_k) - x_{k+1}                 local defect (k = 0..T-1)
    J_k   = Df(x_k)                           reference-path Jacobian
    Phi_k = J_{T-1} ... J_{k+1}              tail propagator (Phi_{T-1} = I)
    q_k   = Phi_k e_k,   dlin = sum_k q_k     first-order forced error

The realised error E = ||f^T(x_0) - x_T|| factorises exactly as

    E = S_e * gbar * eta * C * N                                     (*)

    S_e  = sum_k ||e_k||                            defect magnitude
    gbar = B / S_e,   B = sum_k s1(Phi_k) ||e_k||   defect-weighted worst-case gain
    eta  = S_q / B,   S_q = sum_k ||q_k||           directional efficiency
    C    = ||dlin|| / S_q                           vector cancellation
    N    = E / ||dlin||                             nonlinear factor (1/R_lin)

Null models for the two attenuation factors:

    isotropic   E_u ||Phi u||^2 = ||Phi||_F^2 / d            (u uniform on the sphere)
    covariance  E_S ||Phi e||^2 / E_S||e||^2 = tr(Phi S Phi^T) / tr S
    incoherent  C_inc = sqrt(sum_k ||q_k||^2) / S_q = T_eff^{-1/2}

with S the defect second-moment matrix estimated on independent data.

The closed forms above are mean-square (RMS) gains.  The efficiency eta is a
ratio of sums of norms, and by Jensen's inequality E||Phi u|| < (E||Phi u||^2)^{1/2},
so comparing eta with an RMS null biases the ratio below one (by up to
sqrt(2/pi) ~ 0.8 when Phi has rank one).  The orientation ratio is therefore
computed against Monte Carlo norm gains, E||Phi u|| / ||u||, with u drawn from
N(0, S) (``eta_covmc``) or isotropically (``eta_isomc``).

Coherence null for temporally independent defects: if the e_k were drawn
independently from a distribution with mean m and covariance S - m m^T,

    E ||dlin||^2 = sum_k tr(Phi_k (S - m m^T) Phi_k^T) + ||sum_k Phi_k m||^2,
    E sum_k ||q_k||^2 = sum_k tr(Phi_k S Phi_k^T),

and kappa_iid = (E||dlin||^2 / E sum||q_k||^2)^{1/2} is the coherence produced by
the defect mean (bias) alone.  kappa / kappa_iid measures the coherence due to
the temporal correlation of the defects along the trajectory.
"""
from __future__ import annotations
import numpy as np


def rollout(f, x0, T):
    xs = np.empty((T + 1, x0.shape[-1]))
    xs[0] = x0
    for k in range(T):
        xs[k + 1] = f(xs[k])
    return xs


def defect_second_moment(f, states, return_mean=False):
    """S = mean_k e_k e_k^T over consecutive reference states (and the mean defect)."""
    D = f(states[:-1]) - states[1:]
    S = D.T @ D / len(D)
    return (S, D.mean(0)) if return_mean else S


def _spectral_stats(p):
    cum = np.cumsum(p)
    return dict(p1=float(p[0]), P3=float(cum[min(2, len(p) - 1)]), P5=float(cum[min(4, len(p) - 1)]),
                m90=int(np.searchsorted(cum, 0.9) + 1), npart=float(1.0 / np.sum(p * p)))


def _mc_null_stats(S, Vt, U, prefix):
    """Monte Carlo spectral statistics for random defects u (rows of U) under a tail with SVD (S, Vt)."""
    C = U @ Vt.T                                   # coefficients in the right singular basis
    energy = (S[None, :] * C) ** 2
    p = energy / energy.sum(1, keepdims=True)
    cum = np.cumsum(p, axis=1)
    un = np.linalg.norm(U, axis=1)
    return {f'p1_{prefix}': float(np.mean(p[:, 0])), f'P3_{prefix}': float(np.mean(cum[:, min(2, p.shape[1] - 1)])),
            f'P5_{prefix}': float(np.mean(cum[:, min(4, p.shape[1] - 1)])),
            f'm90_{prefix}': float(np.mean(np.argmax(cum >= 0.9, axis=1) + 1)),
            f'npart_{prefix}': float(np.mean(1.0 / np.sum(p * p, axis=1))),
            f'alignment_{prefix}': float(np.mean(np.abs(C[:, 0]) / un)),
            f'efficiency_{prefix}': float(np.mean(np.sqrt(energy.sum(1)) / (S[0] * un)))}


def window_diagnostics(f, true_states, jacs, Sigma=None, spectrum=True, model_states=None,
                       null_draws=16, rng=None, mean_defect=None, norm_draws=64):
    """All window-level quantities and per-term records.

    true_states : (T+1, d) reference states in the surrogate's coordinates
    jacs        : (T, d, d) Df(x_k), k = 0..T-1, on the reference path
    Sigma       : (d, d) defect second moment for the covariance null (optional)
    spectrum    : compute full singular spectra of every tail propagator
    """
    T = len(true_states) - 1
    d = true_states.shape[1]
    if model_states is None:
        model_states = rollout(f, true_states[0], T)
    delta = model_states[-1] - true_states[-1]
    E = float(np.linalg.norm(delta))
    path_err = np.linalg.norm(model_states[1:] - true_states[1:], axis=1)
    defects = f(true_states[:-1]) - true_states[1:]
    trS = float(np.trace(Sigma)) if Sigma is not None else np.nan
    rng = rng if rng is not None else np.random.default_rng(12345)
    if spectrum and null_draws:
        U_iso = rng.standard_normal((null_draws, d))
        if Sigma is not None:
            Lc = np.linalg.cholesky(Sigma + 1e-12 * trS / d * np.eye(d))
            U_cov = rng.standard_normal((null_draws, d)) @ Lc.T
    # unit-norm draws for the Monte Carlo norm gains (common random numbers for all tails)
    rng_n = np.random.default_rng(2024)
    V_iso = rng_n.standard_normal((norm_draws, d))
    V_iso /= np.linalg.norm(V_iso, axis=1, keepdims=True)
    if Sigma is not None:
        Lc_n = np.linalg.cholesky(Sigma + 1e-12 * trS / d * np.eye(d))
        V_cov = rng_n.standard_normal((norm_draws, d)) @ Lc_n.T
        V_cov /= np.linalg.norm(V_cov, axis=1, keepdims=True)
    if Sigma is not None and mean_defect is not None:
        mu = np.asarray(mean_defect, float)
        Ccen = Sigma - np.outer(mu, mu)
        m_acc = np.zeros(d)

    P = np.eye(d)
    dlin = np.zeros(d)
    W = np.zeros((d, d))
    terms = []
    acc = dict(B=0.0, Sq=0.0, Se=0.0, q2=0.0, Biso=0.0, Bcov=0.0, e2iso=0.0, e2cov=0.0,
               Bisomc=0.0, Bcovmc=0.0, trS_sum=0.0, trC_sum=0.0)
    for k in range(T - 1, -1, -1):
        e = defects[k]
        q = P @ e
        dlin += q
        en = float(np.linalg.norm(e)); qn = float(np.linalg.norm(q))
        if spectrum:
            _, S, Vt = np.linalg.svd(P)
        else:
            S = np.linalg.svd(P, compute_uv=False); Vt = None
        s1 = float(S[0])
        giso2 = float(np.sum(S ** 2)) / d
        acc['B'] += s1 * en; acc['Sq'] += qn; acc['Se'] += en; acc['q2'] += qn * qn
        acc['Biso'] += np.sqrt(giso2) * en; acc['e2iso'] += giso2 * en * en
        acc['Bisomc'] += float(np.mean(np.linalg.norm(V_iso @ P.T, axis=1))) * en
        rec = dict(k=k, tail_len=T - 1 - k, sigma_tail=s1, defect_norm=en, propagated_norm=qn,
                   gain=qn / en if en > 0 else np.nan, gain_iso=np.sqrt(giso2),
                   srank=float(np.sum(S ** 2) / S[0] ** 2))
        if Sigma is not None:
            gcov2 = float(np.einsum('ij,jk,ik->', P, Sigma, P)) / trS
            acc['Bcov'] += np.sqrt(gcov2) * en; acc['e2cov'] += gcov2 * en * en
            gcovmc = float(np.mean(np.linalg.norm(V_cov @ P.T, axis=1)))
            acc['Bcovmc'] += gcovmc * en
            rec['gain_cov'] = np.sqrt(gcov2)
            rec['gain_covmc'] = gcovmc
            if mean_defect is not None:
                acc['trS_sum'] += gcov2 * trS
                acc['trC_sum'] += float(np.einsum('ij,jk,ik->', P, Ccen, P))
                m_acc += P @ mu
        identity = (S[0] - S[-1]) < 1e-10
        if spectrum and en > 0 and not identity:
            c = Vt @ e
            energy = (S * c) ** 2
            p = energy / energy.sum()
            rec['alignment'] = abs(c[0]) / en if S[0] - S[1] > 1e-10 else np.nan
            rec['efficiency'] = qn / (s1 * en)
            rec.update(_spectral_stats(p))
            p_iso = S ** 2 / np.sum(S ** 2)              # expected profile for isotropic defects
            rec.update({f'{key}_iso': val for key, val in _spectral_stats(p_iso).items()})
            if Sigma is not None:
                w = np.einsum('ij,jk,ik->i', Vt, Sigma, Vt)   # v_i^T S v_i
                p_cov = S ** 2 * w / np.sum(S ** 2 * w)
                rec.update({f'{key}_cov': val for key, val in _spectral_stats(p_cov).items()})
            if null_draws:
                rec.update(_mc_null_stats(S, Vt, U_iso, 'isoMC'))
                if Sigma is not None:
                    rec.update(_mc_null_stats(S, Vt, U_cov, 'covMC'))
        terms.append(rec)
        W += P @ P.T
        P = P @ jacs[k]
    terms.reverse()
    s_full = np.linalg.svd(P, compute_uv=False)          # P = J_{T-1}...J_0 = Phi(T,0)
    Ln = float(np.linalg.norm(dlin))
    J0 = jacs[0]
    lam_W = np.linalg.eigvalsh(W)
    B, Sq, Se, q2 = acc['B'], acc['Sq'], acc['Se'], acc['q2']
    out = dict(
        T=T, E=E, E_rel=E / float(np.linalg.norm(true_states[-1])), Lin=Ln,
        Rlin=Ln / E if E > 0 else np.nan, lin_vec_err=float(np.linalg.norm(dlin - delta)),
        Se=Se, B=B, Sq=Sq, q2=q2,
        gbar=B / Se, eta=Sq / B, C=Ln / Sq, N=E / Ln,
        eta_iso=acc['Biso'] / B, alpha_iso=np.sqrt(q2 / acc['e2iso']),
        eta_isomc=acc['Bisomc'] / B, orient_iso=(Sq / acc['Bisomc']),
        C_inc=np.sqrt(q2) / Sq, kappa=Ln / np.sqrt(q2), Teff=Sq * Sq / q2,
        Gmax=float(s_full[0] ** 2), sigma1_full=float(s_full[0]),
        sigma1_tail0=terms[0]['sigma_tail'], forced_gain=float(np.sqrt(max(lam_W[-1], 0.0))),
        gramian_trace=float(np.trace(W)),
        rhoJ=float(np.max(np.abs(np.linalg.eigvals(J0)))), sigma1J=float(np.linalg.svd(J0, compute_uv=False)[0]),
        froJ=float(np.linalg.norm(J0)), one_step=float(np.linalg.norm(defects[0])),
        path_err_max=float(path_err.max()), path_err_mean=float(path_err.mean()),
    )
    if Sigma is not None:
        out.update(eta_cov=acc['Bcov'] / B, alpha_cov=np.sqrt(q2 / acc['e2cov']),
                   eta_covmc=acc['Bcovmc'] / B, orient=Sq / acc['Bcovmc'])
        if mean_defect is not None:
            EL2 = acc['trC_sum'] + float(m_acc @ m_acc)
            out.update(kappa_iid=float(np.sqrt(EL2 / acc['trS_sum'])),
                       bias_frac=float(m_acc @ m_acc / EL2))
    return out, terms


# ------------------------------------------------------------------------
# variance attribution
# ------------------------------------------------------------------------

FACTORS_5 = ('Se', 'gbar', 'eta', 'C', 'N')


def add_log_factors(df):
    """Adds the exact log-factors of (*) and the null-split factors.

    log E = log Se + log gbar + log eta + log C + log N
    eta   = eta_cov * (eta / eta_cov)      (geometry of the defect covariance x state-specific orientation)
    C     = C_inc * kappa                  (number of contributing injections x temporal coherence)
    """
    out = df.copy()
    for f in FACTORS_5:
        out['l_' + f] = np.log(out[f])
    out['l_E'] = np.log(out['E'])
    if 'eta_covmc' in out:
        out['l_eta_cov'] = np.log(out['eta_covmc'])          # Monte Carlo norm null (unbiased)
        out['l_orient'] = out['l_eta'] - out['l_eta_cov']
    elif 'eta_cov' in out:                                     # legacy outputs (RMS null, biased)
        out['l_eta_cov'] = np.log(out['eta_cov'])
        out['l_orient'] = out['l_eta'] - out['l_eta_cov']
    out['l_C_inc'] = np.log(out['C_inc'])
    out['l_kappa'] = np.log(out['kappa'])
    if 'kappa_iid' in out:
        out['l_kappa_iid'] = np.log(out['kappa_iid'])
        out['l_kappa_corr'] = out['l_kappa'] - out['l_kappa_iid']
    return out


def covariance_shares(df, parts, target='l_E', within='seed'):
    """share_i = Cov(log E, log X_i) / Var(log E); sums to one when sum_i log X_i = log E.

    With ``within='seed'`` every log-factor is centred within its seed first, so
    the shares describe the variability across initial states of a given model
    and are not inflated by accuracy differences between models.
    """
    cols = [target] + list(parts)
    X = df[cols].copy()
    if within is not None and within in df:
        X = X - X.groupby(df[within]).transform('mean')
    y = X[target].to_numpy()
    v = np.var(y, ddof=1)
    return {p: float(np.cov(y, X[p].to_numpy(), ddof=1)[0, 1] / v) for p in parts}
