"""Dynamical fidelity of the surrogates.

For every model: leading Lyapunov exponents (Benettin/QR) of the learned map
(i) along the reference trajectory, J_k = Df(x_k), and (ii) along its own free
rollout, plus the RMS amplitude of the normalised state along the free rollout
(the reference attractor has unit RMS amplitude in the L96 normalisation and
approximately unit amplitude in the KS normalisation).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from run_campaign_v2 import prepare


def lyap_along(jvp, states, n_exp, dt, transient=200, seed=0):
    rng = np.random.default_rng(seed)
    d = states.shape[1]
    Q = np.linalg.qr(rng.standard_normal((d, n_exp)))[0]
    acc = np.zeros(n_exp); n = 0
    for k, x in enumerate(states[:-1]):
        W = jvp(x, Q.T).T
        Q, R = np.linalg.qr(W)
        s = np.sign(np.diag(R)); s[s == 0] = 1; Q = Q * s
        if k >= transient:
            acc += np.log(np.abs(np.diag(R))); n += 1
    return acc / (n * dt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["l96", "ks"], required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-exp", type=int, default=6)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--norm-from", default=None)
    a = p.parse_args()
    dt = 0.01 if a.system == 'l96' else 0.25
    out = {}
    cache = {}
    for s in [int(v) for v in a.seeds.split(",")]:
        nf = a.norm_from.format(seed=s) if a.norm_from else None
        m, test, _, _ = prepare(a.system, a.models.format(seed=s), s, cache, nf)
        ref = test[:a.steps + 1]
        lam_ref = lyap_along(m.jvp, ref, a.n_exp, dt, seed=s)
        free = np.empty((a.steps + 1, test.shape[1])); free[0] = test[0]
        blew = None
        for k in range(a.steps):
            free[k + 1] = m(free[k])
            if not np.all(np.isfinite(free[k + 1])) or np.linalg.norm(free[k + 1]) > 1e3:
                blew = k + 1; break
        if blew is None:
            lam_free = lyap_along(m.jvp, free, a.n_exp, dt, seed=s)
            amp = np.sqrt(np.mean(free ** 2, axis=1))
        else:
            lam_free = [float('nan')] * a.n_exp
            amp = np.sqrt(np.mean(free[:blew] ** 2, axis=1))
        amp_ref = np.sqrt(np.mean(ref ** 2, axis=1))
        out[str(s)] = dict(lyap_reference_path=np.asarray(lam_ref).tolist(), lyap_free_run=np.asarray(lam_free).tolist(),
                           blow_up_step=blew, amp_free_mean=float(amp[len(amp) // 2:].mean()),
                           amp_free_final=float(amp[-1]), amp_ref_mean=float(amp_ref.mean()),
                           amp_free_curve=amp[::50].tolist())
        print(a.label, s, np.round(lam_ref[:3], 3), np.round(np.asarray(lam_free)[:3], 3), 'amp', round(out[str(s)]['amp_free_mean'], 3), 'ref', round(out[str(s)]['amp_ref_mean'], 3), 'blow', blew, flush=True)
    od = Path("results/current/fidelity"); od.mkdir(parents=True, exist_ok=True)
    (od / f"{a.label}.json").write_text(json.dumps(out, indent=1))
