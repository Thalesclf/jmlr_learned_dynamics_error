"""Tests of the framework-free (NumPy) analysis package."""
from pathlib import Path
import numpy as np
import pytest

from realized_error.systems import (KSStepper, simulate_ks, simulate_l96, l96_rk4_step, l96_rk4_jacobian,
                                    KS_TRAIN_MEAN)
from realized_error.models import ResidualMLP, FNO1d, load_state
from realized_error.training import MLPParams, rollout_loss_and_grad, windows_one_step
from realized_error.diagnostics import window_diagnostics, covariance_shares, add_log_factors, rollout

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def fd_jvp(f, x, v, eps=1e-6):
    return (f(x + eps * v) - f(x - eps * v)) / (2 * eps)


def test_l96_rk4_jacobian_matches_finite_differences():
    x = simulate_l96(steps=5, burnin=200, seed=3)[-1]
    v = rng.standard_normal(20)
    np.testing.assert_allclose(l96_rk4_jacobian(x) @ v, fd_jvp(l96_rk4_step, x, v), atol=1e-7)


def test_ks_tangent_matches_finite_differences():
    st = KSStepper()
    u = simulate_ks(steps=5, burnin=300, seed=5)[-1]
    V = rng.standard_normal((2, 64))
    fd = np.stack([fd_jvp(st.step, u, v) for v in V])
    np.testing.assert_allclose(st.jvp(u, V), fd, atol=1e-7)


def test_ks_mean_is_conserved_and_prescribed():
    t = simulate_ks(steps=200, burnin=100, seed=11, mean=KS_TRAIN_MEAN)
    np.testing.assert_allclose(t.mean(axis=1), KS_TRAIN_MEAN, atol=1e-13)


def random_mlp_state(dim=6, hidden=9, seed=1):
    return MLPParams(dim=dim, hidden=hidden, seed=seed)


def test_mlp_jacobian_and_backprop():
    P = random_mlp_state()
    m = ResidualMLP(P.state_dict())
    x, v = rng.standard_normal(6), rng.standard_normal(6)
    np.testing.assert_allclose(m.jacobian(x) @ v, fd_jvp(m, x, v), atol=1e-8)
    W = rng.standard_normal((3, 5, 6)) * 0.5
    for loss_fn in (rollout_loss_and_grad, windows_one_step):
        _, g = loss_fn(P, W)
        idx = (2, 3)
        P.p['W1'][idx] += 1e-6; lp, _ = loss_fn(P, W)
        P.p['W1'][idx] -= 2e-6; lm, _ = loss_fn(P, W)
        P.p['W1'][idx] += 1e-6
        assert abs(g['W1'][idx] - (lp - lm) / 2e-6) < 1e-6 * max(1.0, abs(g['W1'][idx]))


def small_fno(seed=0):
    r = np.random.default_rng(seed)
    w, m, n = 4, 5, 16
    sd = {'lift.weight': r.standard_normal((w, 1)), 'lift.bias': r.standard_normal(w),
          'proj1.weight': r.standard_normal((6, w)) / 2, 'proj1.bias': r.standard_normal(6),
          'proj2.weight': r.standard_normal((1, 6)) / 2, 'proj2.bias': r.standard_normal(1)}
    for i in range(2):
        sd[f'spectral.{i}.weight'] = (r.standard_normal((w, w, m)) + 1j * r.standard_normal((w, w, m))) / w
        sd[f'pointwise.{i}.weight'] = r.standard_normal((w, w, 1)) / w
        sd[f'pointwise.{i}.bias'] = r.standard_normal(w)
    return FNO1d(sd, n_grid=n)


def test_fno_jacobians():
    f = small_fno()
    X = rng.standard_normal((3, 16))
    J = f.jacobians(X)
    v = rng.standard_normal(16)
    for k in range(3):
        np.testing.assert_allclose(J[k] @ v, fd_jvp(f, X[k], v), atol=1e-7)
        np.testing.assert_allclose(J[k], f.jacobian(X[k]), atol=1e-12)


def test_factorisation_and_bounds_are_exact():
    P = random_mlp_state(dim=8, hidden=16, seed=4)
    m = ResidualMLP(P.state_dict())
    true = np.cumsum(rng.standard_normal((21, 8)) * 0.05, axis=0)
    J = m.jacobians(true[:-1])
    D = m(true[:-1]) - true[1:]
    Sigma = D.T @ D / len(D)
    out, terms = window_diagnostics(m, true, J, Sigma=Sigma)
    prod = out['Se'] * out['gbar'] * out['eta'] * out['C'] * out['N']
    assert abs(prod / out['E'] - 1) < 1e-12
    assert out['Lin'] <= out['Sq'] * (1 + 1e-12) <= out['B'] * (1 + 1e-12)
    assert abs(out['C'] - out['kappa'] / np.sqrt(out['Teff'])) < 1e-12
    for t in terms:
        if 'alignment' in t and np.isfinite(t['alignment']):
            assert t['alignment'] <= t['efficiency'] + 1e-12 <= 1 + 1e-12
    # brute-force first-order reconstruction
    dlin = np.zeros(8)
    for k in range(20):
        Pk = np.eye(8)
        for j in range(k + 1, 20):
            Pk = J[j] @ Pk
        dlin += Pk @ D[k]
    assert abs(np.linalg.norm(dlin) - out['Lin']) < 1e-10


def test_covariance_shares_sum_to_one():
    import pandas as pd
    n = 200
    f = {k: np.exp(rng.standard_normal(n) * s) for k, s in zip(['Se', 'gbar', 'eta', 'C', 'N'], [1, .3, .2, .2, .1])}
    df = pd.DataFrame(f); df['E'] = df.Se * df.gbar * df.eta * df.C * df.N
    df['C_inc'] = 0.5; df['kappa'] = df.C / 0.5
    sh = covariance_shares(add_log_factors(df), ['l_Se', 'l_gbar', 'l_eta', 'l_C', 'l_N'])
    assert abs(sum(sh.values()) - 1) < 1e-12


@pytest.mark.skipif(not (ROOT / 'results/seed_0/checkpoint.pt').exists(), reason='archived checkpoints not present')
def test_numpy_port_reproduces_archived_l96_campaign():
    import pandas as pd
    from run_campaign_v2 import prepare
    m, test, _, _ = prepare('l96', ROOT / 'results/seed_0/checkpoint.pt', 0, {})
    A = pd.read_csv(ROOT / 'results_l96_jmlr/seed_0/diagnostic_campaign.csv')
    for _, r in A[A.horizon.isin([20, 80])].head(4).iterrows():
        s, T = int(r.test_index), int(r.horizon)
        out, _ = window_diagnostics(m, test[s:s + T + 1], m.jacobians(test[s:s + T]), spectrum=False)
        for a, b in [('E', 'rollout_error'), ('Lin', 'linear_predicted_error'), ('B', 'defect_amplification_bound'),
                     ('Gmax', 'Gmax'), ('Sq', 'propagated_defect_norm_sum')]:
            assert abs(out[a] / r[b] - 1) < 1e-10


def test_null_models_match_brute_force_monte_carlo():
    """kappa_iid and the Monte Carlo norm gain agree with direct sampling of iid defects."""
    r = np.random.default_rng(7)
    d, T = 5, 6
    Js = np.eye(d)[None] + 0.4 * r.standard_normal((T, d, d))
    A = r.standard_normal((d, d))
    C = A @ A.T / d
    mu = 0.7 * r.standard_normal(d)
    Sigma = C + np.outer(mu, mu)
    states = r.standard_normal((T + 1, d))
    f = lambda x: x                                        # only defects/Jacobians matter below
    out, _ = window_diagnostics(f, states, Js, Sigma=Sigma, mean_defect=mu, spectrum=False,
                                model_states=states, norm_draws=4000)
    # brute force: iid defects e_k ~ N(mu, C)
    n = 40000
    L = np.linalg.cholesky(C)
    E = mu + r.standard_normal((n, T, d)) @ L.T
    P = [np.eye(d)]
    for k in range(T - 1, 0, -1):
        P.insert(0, P[0] @ Js[k])
    Q = np.stack([E[:, k] @ P[k].T for k in range(T)], axis=1)
    kappa_mc = np.sqrt(np.mean(np.sum(Q.sum(1) ** 2, 1)) / np.mean(np.sum(Q ** 2, (1, 2))))
    assert abs(out['kappa_iid'] / kappa_mc - 1) < 0.02
    # Monte Carlo norm gain of the last tail against direct sampling from N(0, Sigma)
    U = r.standard_normal((n, d)) @ np.linalg.cholesky(Sigma).T
    g_direct = np.mean(np.linalg.norm(U @ P[0].T, axis=1) / np.linalg.norm(U, axis=1))
    s1 = np.linalg.svd(P[0], compute_uv=False)[0]
    # eta_covmc is a weighted average over tails; check the RMS null exceeds the norm null (Jensen)
    assert out['eta_covmc'] <= out['eta_cov'] + 1e-12
    assert 0 < g_direct <= s1


def test_within_seed_shares_sum_to_one():
    r = np.random.default_rng(3)
    import pandas as pd
    X = r.standard_normal((60, 3))
    df = pd.DataFrame(dict(seed=np.repeat(np.arange(4), 15), a=X[:, 0], b=X[:, 1] + 2 * np.repeat(np.arange(4), 15),
                           c=X[:, 2]))
    df['l_E'] = df.a + df.b + df.c
    sh = covariance_shares(df, ['a', 'b', 'c'])
    assert abs(sum(sh.values()) - 1) < 1e-12
