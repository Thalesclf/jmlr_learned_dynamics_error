"""Train Lorenz--96 residual-MLP surrogates at several optimisation budgets (NumPy).

The data pipeline is identical to experiments_campaign_spectral_jmlr.py:
trajectory seed = model seed, 20,000 retained RK4 states (dt = 0.01, F = 8,
d = 20), chronological 70/30 split, per-component standardisation with
training statistics.  Only the optimisation budget changes:

    e20    : the archived 20-epoch PyTorch checkpoints (results/seed_*/checkpoint.pt)
    e150   : 150 epochs, batch 256, Adam 1e-3 with cosine decay to 1e-5
    e2000  : 2000 epochs, batch 256, Adam 1e-3 with cosine decay to 1e-5
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from realized_error.systems import simulate_l96
from realized_error.training import MLPParams, train_one_step


def l96_data(seed, steps=20000, burnin=2000):
    traj = simulate_l96(seed=seed, steps=steps, burnin=burnin)
    x = traj[:-1]
    ntr = int(0.7 * len(x))
    mean = x[:ntr].mean(0)
    std = x[:ntr].std(0, ddof=1)
    z = (traj - mean) / std
    return z, ntr, mean, std


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-final", type=float, default=1e-5)
    p.add_argument("--outdir", default="results/current/l96_models")
    a = p.parse_args()
    for seed in [int(s) for s in a.seeds.split(",")]:
        z, ntr, mean, std = l96_data(seed)
        X, Y = z[:-1], z[1:]
        P = MLPParams(dim=20, hidden=128, seed=seed)
        t0 = time.time()
        hist = train_one_step(P, X[:ntr], Y[:ntr], epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                              lr_final=a.lr_final, seed=seed, xva=X[ntr:], yva=Y[ntr:],
                              log_every=max(1, a.epochs // 10))
        out = Path(a.outdir) / a.label / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "model.npz", **P.state_dict(), mean=mean, std=std)
        meta = dict(seed=seed, label=a.label, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                    lr_final=a.lr_final, train_seconds=round(time.time() - t0, 1),
                    test_one_step_mse=hist[-1]['test_mse'], train_mse_last_epoch=hist[-1]['train_mse'],
                    persistence_test_mse=float(np.mean((X[ntr:] - Y[ntr:]) ** 2)), history=hist)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(a.label, seed, "test mse", meta['test_one_step_mse'], f"{meta['train_seconds']}s", flush=True)
