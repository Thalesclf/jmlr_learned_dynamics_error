"""Leading Lyapunov exponents of the reference systems and of the learned surrogates.

Used to express forecast horizons in Lyapunov times (lambda_1 * T * dt).
Writes results/current/lyapunov.json.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from realized_error.systems import (simulate_l96, l96_rk4_step, l96_rk4_jacobian, KSStepper, simulate_ks,
                                    lyapunov_qr)


def true_l96(n_exp, n_steps, seed):
    x0 = simulate_l96(steps=1, burnin=5000, seed=seed)[-1]
    jvp = lambda x, V: V @ l96_rk4_jacobian(x).T
    lam, _ = lyapunov_qr(l96_rk4_step, jvp, x0, n_exp, n_steps, dt=0.01, transient=2000, seed=seed)
    return lam


def true_ks(n_exp, n_steps, seed):
    st = KSStepper()
    x0 = simulate_ks(steps=1, burnin=4000, seed=seed)[-1]
    lam, _ = lyapunov_qr(st.step, st.jvp, x0, n_exp, n_steps, dt=0.25, transient=1000, seed=seed)
    return lam


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["l96", "ks"], required=True)
    p.add_argument("--n-exp", type=int, default=12)
    p.add_argument("--n-steps", type=int, default=100000)
    p.add_argument("--seeds", default="11,12,13")
    p.add_argument("--out", default="results/current/lyapunov_true_{system}.json")
    a = p.parse_args()
    res = {}
    for s in [int(v) for v in a.seeds.split(",")]:
        t0 = time.time()
        lam = true_l96(a.n_exp, a.n_steps, s) if a.system == "l96" else true_ks(a.n_exp, a.n_steps, s)
        res[str(s)] = lam.tolist()
        print(a.system, s, np.round(lam, 4), f"{time.time()-t0:.0f}s", flush=True)
    lam = np.array(list(res.values()))
    out = {"system": a.system, "n_steps": a.n_steps, "per_seed": res,
           "mean": lam.mean(0).tolist(), "std": lam.std(0, ddof=1).tolist() if len(lam) > 1 else None}
    Path(a.out.format(system=a.system)).write_text(json.dumps(out, indent=2))
