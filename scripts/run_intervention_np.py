"""Matched one-step vs direct-rollout fine-tuning of L96 surrogates (NumPy re-implementation).

Follows intervention_campaign.py: both arms start from the same checkpoint,
see the same window stream (default_rng(train_window_seed + seed)), use Adam
with gradient clipping, and the auxiliary rollout term is rescaled by its value
at the initial checkpoint so that lambda is a relative weight:

    L = L_one + lambda * (L_one(0) / R(0)) * R,   R = mean (f^H(x_0) - x_H)^2.

Checkpoints are compared every ``eval_every`` updates on an independent
validation trajectory (mean normalised MSE at the selection horizons); the
initial checkpoint competes.  ``--validation-only`` runs the (H, lambda) sweep
without touching test data; the confirmatory test is run by
run_intervention_eval.py on the saved best checkpoints.
"""
from __future__ import annotations
import argparse, json, itertools
from pathlib import Path
import numpy as np

from realized_error.systems import simulate_l96
from realized_error.training import (MLPParams, Adam, clip_, windows_one_step, rollout_loss_and_grad, add_)
from realized_error.models import load_l96
from run_campaign_v2 import l96_normaliser
from run_l96_models import l96_data


def eval_mse(P, seq, starts, horizons):
    x = seq[starts].copy()
    out = {}
    for h in range(1, max(horizons) + 1):
        x = P(x)
        if h in horizons:
            out[h] = float(np.mean((x - seq[starts + h]) ** 2))
    return out


def finetune(P0, train_seq, val_seq, val_starts, method, H, lam, seed, steps, batch, lr, clip,
             eval_every, select_h, window_seed=20261005, lr_final=None):
    rng = np.random.default_rng(window_seed + seed)
    starts = rng.integers(0, len(train_seq) - H, size=(steps, batch))
    offs = np.arange(H + 1)
    windows = lambda s: train_seq[s[:, None] + offs[None, :]]
    cal = windows(starts[0])
    one0, _ = windows_one_step(P0, cal)
    r0, _ = rollout_loss_and_grad(P0, cal)
    scale = one0 / max(r0, 1e-20)
    P = P0.copy()
    opt = Adam(P.p, lr=lr)
    score = lambda Q: float(np.mean([v for v in eval_mse(Q, val_seq, val_starts, select_h).values()]))
    best, best_step, best_P = score(P), 0, P.copy()
    hist = [dict(step=0, val=best)]
    for it in range(steps):
        W = windows(starts[it])
        l1, g = windows_one_step(P, W)
        if method == 'rollout':
            lr_, gr = rollout_loss_and_grad(P, W)
            add_(g, gr, lam * scale)
        gn = clip_(g, clip)
        cur = lr if lr_final is None else lr_final + 0.5 * (lr - lr_final) * (1 + np.cos(np.pi * it / max(steps - 1, 1)))
        opt.step(P.p, g, lr=cur)
        if (it + 1) % eval_every == 0 or it + 1 == steps:
            v = score(P)
            hist.append(dict(step=it + 1, val=v, grad_norm=gn))
            if v < best:
                best, best_step, best_P = v, it + 1, P.copy()
    return best_P, dict(method=method, H=H, lam=lam, best_val=best, best_step=best_step,
                        scale=scale, history=hist)


def setup(model_path, seed, norm_from, val_seed, val_steps, max_h):
    m, meta = load_l96(model_path)
    mean, std = l96_normaliser(model_path, meta, norm_from)
    z, ntr, _, _ = l96_data(seed)
    z = (simulate_l96(seed=seed) - mean) / std
    train_seq = z[:ntr + 1]
    val = (simulate_l96(seed=val_seed, steps=val_steps, burnin=2000) - mean) / std
    val_starts = np.arange(0, len(val) - max_h - 1, max_h)
    sd = {k: getattr(m, a) for k, a in [('net.0.weight', 'W1'), ('net.0.bias', 'b1'), ('net.2.weight', 'W2'),
                                         ('net.2.bias', 'b2'), ('net.4.weight', 'W3'), ('net.4.bias', 'b3')]}
    return MLPParams(state_dict=sd), train_seq, val, val_starts, mean, std


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True, help="pattern with {seed}")
    p.add_argument("--norm-from", default=None)
    p.add_argument("--label", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--H", default="4,8,16")
    p.add_argument("--lam", default="0.01,0.03,0.1,0.3")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--lr-final", type=float, default=None, help="cosine decay of the learning rate to this value")
    p.add_argument("--eval-every", type=int, default=15)
    p.add_argument("--select-h", default="16,32")
    p.add_argument("--val-seed", type=int, default=20261018)
    p.add_argument("--val-steps", type=int, default=4000)
    p.add_argument("--validation-only", action="store_true")
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    Hs = [int(h) for h in a.H.split(",")]; lams = [float(v) for v in a.lam.split(",")]
    select_h = [int(h) for h in a.select_h.split(",")]
    out = Path("results/current/intervention_np") / a.label
    out.mkdir(parents=True, exist_ok=True)
    data = {s: setup(a.models.format(seed=s), s, a.norm_from.format(seed=s) if a.norm_from else None,
                     a.val_seed, a.val_steps, 64) for s in seeds}
    (out / "protocol.json").write_text(json.dumps(vars(a), indent=1))
    if a.validation_only:
        recs = []
        for H, lam in itertools.product(Hs, lams):
            for s in seeds:
                P0, tr, val, vs, _, _ = data[s]
                _, info = finetune(P0, tr, val, vs, 'rollout', H, lam, s, a.steps, a.batch, a.lr, a.clip,
                                   a.eval_every, select_h, lr_final=a.lr_final)
                recs.append(dict(seed=s, H=H, lam=lam, best_val=info['best_val'], best_step=info['best_step']))
            sub = [r for r in recs if r['H'] == H and r['lam'] == lam]
            print(f"H={H:2d} lam={lam:<5} mean best val {np.mean([r['best_val'] for r in sub]):.5e}", flush=True)
        (out / "validation_sweep.json").write_text(json.dumps(recs, indent=1))
        means = {}
        for r in recs:
            means.setdefault((r['H'], r['lam']), []).append(r['best_val'])
        H, lam = min(means, key=lambda k: np.mean(means[k]))
        (out / "selected.json").write_text(json.dumps(dict(H=H, lam=lam), indent=1))
        print("selected", H, lam)
    else:
        sel = json.loads((out / "selected.json").read_text())
        for s in seeds:
            P0, tr, val, vs, mean, std = data[s]
            od = out / f"seed_{s}"; od.mkdir(exist_ok=True)
            infos = []
            for method in ('one_step', 'rollout'):
                P, info = finetune(P0, tr, val, vs, method, sel['H'], sel['lam'], s, a.steps, a.batch, a.lr,
                                   a.clip, a.eval_every, select_h, lr_final=a.lr_final)
                np.savez(od / f"best_{method}.npz", **P.state_dict(), mean=mean, std=std)
                infos.append(info)
            (od / "results.json").write_text(json.dumps(infos, indent=1))
            print(s, [(i['method'], i['best_step'], round(i['best_val'], 7)) for i in infos], flush=True)
