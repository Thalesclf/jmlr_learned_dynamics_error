"""Controlled, checkpoint-matched defect-aware intervention experiment.

Run from the research project folder containing ks_fno_campaign.py and
experiments_campaign_spectral_jmlr.py. No modification to either baseline file.

Five supported arms start from the SAME trained model checkpoint:
    one_step       L_1
    local_jacobian L_1 + lambda * normalized E_v ||Df(x)v||^2/d
    rollout        L_1 + lambda * normalized ||f^H(x_t)-x_{t+H}||^2/d
    realized       L_1 + lambda * normalized ||delta_H^lin||^2/d
    defect_energy  L_1 + lambda * normalized mean_k ||Phi(H,k+1)e_k||^2/d

where delta_{k+1}^lin = Df(x_k) delta_k^lin + [f(x_k)-x_{k+1}], delta_0=0,
and all Jacobians are evaluated on true, teacher-forced trajectories.

All arms use identical pretrained weights, training windows, optimizer settings,
validation/test trajectories, and checkpoint selection. The realized penalty
uses differentiable JVPs and DOES NOT detach the Jacobian or propagated errors.

The rollout arm is essential: if a defect-aware penalty merely reproduces standard
multistep training, we should say so rather than claim a new method. The
defect_energy arm operationalizes the covariance-weighted quantity discussed in
the manuscript while removing cross terms between different injected defects,
so destructive cancellation cannot make the regularizer artificially small.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Resolve original research scripts relative to this file OR the working directory.
_here = Path(__file__).resolve().parents[1] / "scripts"
for folder in [str(_here), str(Path.cwd())]:
    if folder not in sys.path:
        sys.path.insert(0, folder)

import ks_fno_campaign as ks
import experiments_campaign_spectral_jmlr as l96

METHODS = ("one_step", "local_jacobian", "rollout", "realized", "defect_energy")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model_and_data(args):
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = saved["args"]
    dtype = next(iter(saved["model_state_dict"].values())).real.dtype
    device = torch.device(args.device)

    if args.system == "ks":
        n = int(cfg["n_grid"])
        model = ks.FNO1d(
            n_grid=n, modes=int(cfg["modes"]), width=int(cfg["width"]),
            depth=int(cfg["depth"]), projection_width=int(cfg["projection_width"]),
            outer_residual=bool(cfg["outer_residual"]), dtype=dtype,
        )
        mean = torch.as_tensor(saved["normalizer_mean"], dtype=dtype, device=device)
        std = torch.as_tensor(saved["normalizer_std"], dtype=dtype, device=device)
        train_seed = int(cfg["data_seed"])
        generator_kwargs = {
            "n_grid": n, "length": float(cfg["length"]), "dt": float(cfg["dt"]),
            "sample_stride": int(cfg["sample_stride"]),
        }
        burnin = int(cfg["burnin"])
        train_steps = int(cfg["data_steps"])
        n_train = int(float(cfg["train_fraction"]) * train_steps)
        def simulate(steps, seed):
            return ks.simulate_ks(steps=steps, burnin=burnin, seed=seed, **generator_kwargs)
    else:
        n = int(cfg["dim"])
        model = l96.ResidualMLP(n, int(cfg["hidden"]))
        mean = torch.as_tensor(saved["mean"], dtype=dtype, device=device)
        std = torch.as_tensor(saved["std"], dtype=dtype, device=device)
        train_seed = int(cfg["seed"])
        generator_kwargs = {"n": n, "forcing": float(cfg["forcing"]), "dt": float(cfg["dt"])}
        burnin = int(cfg["burnin"])
        train_steps = int(cfg["data_steps"])
        n_train = int(0.7 * (train_steps - 1))
        def simulate(steps, seed):
            return l96.simulate_l96(steps=steps, burnin=burnin, seed=seed, **generator_kwargs)

    model.load_state_dict(saved["model_state_dict"])
    model.to(device)
    model.eval()
    def to_normalized(data: np.ndarray):
        return (torch.as_tensor(data, dtype=dtype, device=device) - mean) / std

    # Pretrained model used first 70% of original trajectory. Reuse exactly those
    # states for fine-tuning; independent val/test trajectories avoid test leakage.
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    def trajectory(kind: str, steps: int, seed: int):
        # Include every KS/L96 discretization parameter in this cache key.
        import hashlib
        key = json.dumps({"system":args.system,"seed":seed,"steps":steps,
                          "burnin":burnin,"params":generator_kwargs},sort_keys=True)
        suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
        fn = cache_dir / f"{args.system}_{kind}_{suffix}.npz"
        if fn.exists():
            arr = np.load(fn)["trajectory"]
        else:
            print(f"Generating {args.system} {kind}: seed={seed}, steps={steps}", flush=True)
            arr = simulate(steps=steps, seed=seed)
            np.savez_compressed(fn, trajectory=arr)
        return to_normalized(arr)

    original = trajectory("train", train_steps, train_seed)
    # L96 simulate_l96 returns steps states; KS returns steps+1 states.
    train_sequence = original[: n_train + 1]
    val_sequence = trajectory("validation", args.eval_steps, args.val_data_seed)
    test_sequence = None if args.validation_only else trajectory("test", args.eval_steps, args.test_data_seed)
    if len(train_sequence) < args.horizon + 2:
        raise ValueError("Insufficient train data for requested training horizon")
    return model, train_sequence, val_sequence, test_sequence, cfg, float(mean.mean()), float(std.mean())


def select_windows(sequence: torch.Tensor, starts: np.ndarray, horizon: int) -> torch.Tensor:
    # Shape B, H+1, spatial DOF. All windows stay inside train partition.
    offsets = torch.arange(horizon + 1, device=sequence.device)
    indices = torch.as_tensor(starts, dtype=torch.long, device=sequence.device)
    return sequence[indices[:, None] + offsets[None, :]]


def one_step_loss(model, windows):
    B, H_plus, d = windows.shape
    xx = windows[:, :-1, :].reshape(-1, d)
    yy = windows[:, 1:, :].reshape(-1, d)
    return (model(xx) - yy).square().mean()


def forced_linear_error(model, windows):
    """Exact differentiable first-order forced error recurrence along true states.

    JVP computes Df(x_k) delta_k; both f and Jacobian depend on parameters.
    No detach, hence gradients include derivatives through Jacobians.
    """
    B, H_plus, d = windows.shape
    if H_plus < 2:
        raise ValueError("Need at least one prediction step")
    delta = torch.zeros_like(windows[:, 0])
    for k in range(H_plus - 1):
        xk = windows[:, k]
        target = windows[:, k + 1]
        yk, J_delta = torch.func.jvp(model, (xk,), (delta,))
        delta = J_delta + (yk - target)
    return delta



def propagated_defect_energy(model, windows):
    """Mean energy of individually propagated model defects.

    For defect e_k = f(x_k)-x_{k+1} and tail propagator
    Phi(H,k+1), this computes

        (1/H) sum_k mean_batch,dim |Phi(H,k+1)e_k|^2.

    Unlike ||sum_k Phi_k e_k||^2, this quantity contains no cross terms
    between injection times, so cancellation cannot hide large propagated
    contributions. It is a sample analogue of a defect-covariance-weighted
    finite-time sensitivity objective. All JVPs remain differentiable with
    respect to model parameters.
    """
    B, H_plus, d = windows.shape
    H = H_plus - 1
    if H < 1:
        raise ValueError("Need at least one prediction step")
    energies = []
    for k in range(H):
        xk = windows[:, k]
        target = windows[:, k + 1]
        q = model(xk) - target
        # Propagate the kth defect from k+1 to common final time H along
        # the teacher-forced reference trajectory.
        for j in range(k + 1, H):
            xj = windows[:, j]
            _, q = torch.func.jvp(model, (xj,), (q,))
        energies.append(q.square().mean())
    return torch.stack(energies).mean()

def local_jacobian_penalty(model, windows, noise):
    x = windows[:, 0]
    _, Jv = torch.func.jvp(model, (x,), (noise,))
    return Jv.square().mean()


def final_rollout_error(model, windows):
    x = windows[:, 0]
    for _ in range(windows.shape[1] - 1):
        x = model(x)
    return x - windows[:, -1]


def penalty(model, windows, method, noise=None):
    if method == "one_step":
        return windows.new_zeros(())
    if method == "realized":
        return forced_linear_error(model, windows).square().mean()
    if method == "rollout":
        return final_rollout_error(model, windows).square().mean()
    if method == "defect_energy":
        return propagated_defect_energy(model, windows)
    if method == "local_jacobian":
        if noise is None:
            raise ValueError("local_jacobian needs Rademacher noise")
        return local_jacobian_penalty(model, windows, noise)
    raise ValueError(f"Unknown method {method}")


def eval_starts(n_states: int, max_h: int, count: int) -> np.ndarray:
    n_possible = n_states - max_h
    if n_possible < 1:
        raise ValueError("Evaluation trajectory shorter than the maximum horizon")
    # Equal spacing and deterministic; if the trajectory is long enough, avoid
    # overlapping max-horizon windows (reduces pseudoreplication).
    return np.linspace(0, n_possible - 1, min(count, max(1, n_possible // max_h)), dtype=int)


def evaluate_model(model, sequence, horizons, starts, norm_mean, norm_std):
    """Pure open-loop evaluation, with no training-time teacher forcing."""
    model.eval()
    result = {}
    with torch.no_grad():
        first = sequence[:-1]
        result["one_step_mse"] = float((model(first) - sequence[1:]).square().mean())
        x0 = sequence[torch.as_tensor(starts, dtype=torch.long, device=sequence.device)]
        x = x0
        for h in range(1, max(horizons) + 1):
            x = model(x)
            if h in horizons:
                true = sequence[torch.as_tensor(starts + h, dtype=torch.long, device=sequence.device)]
                err = (x - true) * norm_std
                phys_true = true * norm_std + norm_mean
                numerator = err.square().sum(dim=-1)
                denominator = phys_true.square().sum(dim=-1).clamp_min(1e-20)
                rel2 = numerator / denominator
                result[f"h{h}_mse"] = float((x - true).square().mean())
                result[f"h{h}_rel_rmse"] = float(torch.sqrt(rel2.mean()))
                result[f"h{h}_median_rel_error"] = float(torch.median(torch.sqrt(rel2)))
    return result


def save_csv(records, path: Path):
    if not records: return
    fields = list(dict.fromkeys(k for r in records for k in r.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)


def experiment(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch has no CUDA; choose --device cpu")
    if args.val_data_seed == args.test_data_seed:
        raise ValueError("Validation and test trajectory seeds must differ")
    if args.horizon <= 0: raise ValueError("Training horizon must be positive")
    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    if any(m not in METHODS for m in requested):
        raise ValueError(f"Choose methods from {METHODS}")
    eval_h = sorted(set(int(i) for i in args.eval_horizons.split(",")))
    select_h = [int(i) for i in args.selection_horizons.split(",")]
    if not set(select_h).issubset(eval_h):
        raise ValueError("selection-horizons must be included in eval-horizons")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    set_seed(112233 + args.seed)
    base_model, train_seq, val_seq, test_seq, cfg, mean, std = make_model_and_data(args)
    print(f"{args.system} | seed {args.seed} | torch {torch.__version__} | device {args.device}", flush=True)
    print(f"Train {len(train_seq)}, val {len(val_seq)}, test {len(test_seq) if test_seq is not None else '(withheld)'}", flush=True)
    print(f"Methods: {requested} | steps: {args.steps} | H: {args.horizon}", flush=True)
    val_idx = eval_starts(len(val_seq), max(eval_h), args.val_windows)
    test_idx = eval_starts(len(test_seq), max(eval_h), args.test_windows) if test_seq is not None else np.array([], dtype=int)
    base_weights = copy.deepcopy(base_model.state_dict())

    # Fixed random window stream common to all four arms.
    rng = np.random.default_rng(args.train_window_seed + args.seed)
    available = len(train_seq) - args.horizon
    starts = rng.integers(0, available, size=(args.steps, args.batch_size), endpoint=False)
    calibration_starts = starts[0] if args.steps > 0 else np.arange(args.batch_size)
    cal_windows = select_windows(train_seq, calibration_starts, args.horizon)
    # Noise is a unit-variance Rademacher vector; ||Jv||^2/d estimates ||J||_F^2/d.
    noise_rng = np.random.default_rng(args.train_window_seed + 9347 + args.seed)
    noise_arr = noise_rng.integers(0, 2, size=(args.steps + 1, args.batch_size, train_seq.shape[-1]), dtype=np.int8)
    noise_arr = (noise_arr.astype("float64") * 2 - 1)
    cal_noise = torch.as_tensor(noise_arr[0], device=train_seq.device, dtype=train_seq.dtype)
    noise = [torch.as_tensor(arr, device=train_seq.device, dtype=train_seq.dtype) for arr in noise_arr[1:]]

    with torch.no_grad():
        initial_one = float(one_step_loss(base_model, cal_windows))
    reg_initial = {}
    for method in requested:
        if method == "one_step": continue
        # Calibration is detached before fitting; scales are fixed for all updates.
        with torch.enable_grad():
            reg_initial[method] = float(penalty(base_model, cal_windows, method, cal_noise).detach())
    scales = {method: initial_one / max(value, 1e-20) for method,value in reg_initial.items()}
    lambda_weights = {m: args.lambda_weight for m in METHODS}
    if args.lambda_by_method:
        for entry in args.lambda_by_method.split(","):
            method, value = entry.split("=", 1)
            method = method.strip()
            if method not in METHODS: raise ValueError(f"Unknown method in --lambda-by-method: {method}")
            lambda_weights[method] = float(value)
    print(f"Initial one-step: {initial_one:.4e}; regularizer scales: {scales}; weights: {lambda_weights}", flush=True)

    config = vars(args).copy()
    config.update(source_checkpoint=str(Path(args.checkpoint)), pretrained_config=cfg,
                  original_normalizer_mean=mean, original_normalizer_std=std,
                  normalized_penalty_scales=scales, resolved_lambda_weights=lambda_weights, validation_starts=val_idx.tolist(),
                  test_starts=test_idx.tolist(), pytorch_version=torch.__version__,
                  methodology="matched pretrained finetuning, independent validation/test trajectories")
    (outdir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    all_history = []
    outcomes = []
    if test_seq is not None:
        initial = evaluate_model(base_model, test_seq, eval_h, test_idx, mean, std)
        # Initial checkpoint is a common comparator, not an intervention arm.
        print("Initial checkpoint test RMSE:", {h: initial[f"h{h}_rel_rmse"] for h in eval_h}, flush=True)
        (outdir / "initial_model_test.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    for method in requested:
        model = copy.deepcopy(base_model)
        model.load_state_dict(base_weights)
        model.to(args.device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_metric = math.inf
        best_step = 0
        best_weights = None
        histories=[]
        t0 = time.time()
        # Each arm also competes with its common initial state for val selection.
        v0 = evaluate_model(model, val_seq, eval_h, val_idx, mean, std)
        best_metric = float(np.mean([v0[f"h{h}_mse"] for h in select_h]))
        best_weights = copy.deepcopy(model.state_dict())
        histories.append({"method":method,"step":0,"loss":None,"one_step":initial_one,
                          "regularizer":reg_initial.get(method, 0.0),"validation_score":best_metric})
        for step in range(args.steps):
            model.train()
            windows = select_windows(train_seq, starts[step], args.horizon)
            loss_one = one_step_loss(model, windows)
            loss_reg = penalty(model, windows, method, noise[step])
            loss = loss_one + lambda_weights[method] * scales.get(method, 0.0) * loss_reg
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss {method}, step={step + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            if not torch.isfinite(grad):
                raise FloatingPointError(f"Nonfinite gradient {method}, step={step + 1}")
            opt.step()
            if (step+1) % args.eval_every == 0 or step+1 == args.steps:
                v = evaluate_model(model, val_seq, eval_h, val_idx, mean, std)
                metric = float(np.mean([v[f"h{h}_mse"] for h in select_h]))
                if metric < best_metric:
                    best_metric = metric
                    best_step = step + 1
                    best_weights = copy.deepcopy(model.state_dict())
                rec={"method":method,"step":step+1,"loss":float(loss.detach()),
                     "one_step":float(loss_one.detach()),"regularizer":float(loss_reg.detach()),
                     "validation_score":metric,"grad_norm":float(grad)}
                histories.append(rec)
                print(f"{method:>14} {step+1:>4}/{args.steps}: one={rec['one_step']:.3e}, "
                      f"reg={rec['regularizer']:.3e}, val={metric:.3e}, best={best_metric:.3e}", flush=True)

        model.load_state_dict(best_weights)
        test = (evaluate_model(model, test_seq, eval_h, test_idx, mean, std)
                if test_seq is not None else None)
        result={"system":args.system,"seed":args.seed,"method":method,
                "selected_step":best_step,"selection_score":best_metric,
                "elapsed_seconds":round(time.time()-t0,2),"lambda_weight":lambda_weights[method]}
        if test is not None: result["test"] = test
        outcomes.append(result)
        all_history.extend(histories)
        torch.save({"model_state_dict":model.state_dict(),"method":method,
                    "selected_step":best_step,"source_checkpoint":str(args.checkpoint),
                    "normalizer_mean":mean,"normalizer_std":std},
                   outdir / f"best_{method}.pt")
        save_csv(all_history, outdir / "training_history.csv")
        result_name = "results_validation.json" if args.validation_only else "results.json"
        (outdir / result_name).write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
        if test is None:
            print(f"Finished {method}: best step {best_step}, validation score {best_metric:.5g}", flush=True)
        else:
            print(f"Finished {method}: best step {best_step}, test h{eval_h[-1]} relative RMSE "
                  f"{test[f'h{eval_h[-1]}_rel_rmse']:.5g}", flush=True)
    print(f"\nCompleted intervention study: {outdir}", flush=True)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", choices=["ks", "l96"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--lambda-weight", type=float, default=0.5)
    p.add_argument("--lambda-by-method", default="",
        help="e.g. local_jacobian=0.1,rollout=0.05,realized=0.05")
    p.add_argument("--validation-only", action="store_true",
        help="Tune penalty weights on validation, WITHOUT generating or evaluating test data")
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=15)
    p.add_argument("--eval-steps", type=int, default=1200)
    p.add_argument("--eval-horizons", default="1,4,8,16,32,64")
    p.add_argument("--selection-horizons", default="8,16")
    p.add_argument("--val-windows", type=int, default=16)
    p.add_argument("--test-windows", type=int, default=24)
    p.add_argument("--train-window-seed", type=int, default=20261005)
    p.add_argument("--val-data-seed", type=int, default=31415)
    p.add_argument("--test-data-seed", type=int, default=27182)
    p.add_argument("--cache-dir", default="results_intervention/_cache")
    return p

if __name__ == "__main__":
    experiment(build_parser().parse_args())
