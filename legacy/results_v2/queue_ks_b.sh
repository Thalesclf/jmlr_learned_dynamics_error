#!/bin/bash
# core 2: finish ks_pre (already running), then the one-step arm, the long-trajectory evaluation and fidelity
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Written for the layout used until October 2026 (scripts and results_v2/ at the top level of the project).
cd "$(dirname "$0")/.."
until [ -f results_v2/campaign/ks_pre/seed_4/meta.json ]; do sleep 20; done
echo "ks_pre done $(date)"
python3 run_campaign_v2.py --system ks --label ks_one_step --models "results_rollout_retry_final/ks/seed_{seed}/best_one_step.pt" > results_v2/campaign_ks_one.log 2>&1
echo "ks_one_step done $(date)"
python3 run_intervention_eval.py --system ks --root results_rollout_retry_final/ks --label ks_retry --initial "results_ks/seed_{seed}/checkpoint.pt" > results_v2/intervention_eval_ks.log 2>&1
python3 run_fidelity.py --system ks --models "results_ks/seed_{seed}/checkpoint.pt" --label ks_pre > results_v2/fidelity_ks_pre.log 2>&1
echo "QUEUE_KS_B_DONE $(date)"
