#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Written for the layout used until October 2026 (scripts and results_v2/ at the top level of the project).
cd "$(dirname "$0")/.."
python3 run_campaign_v2.py --system ks --label ks_pre --models "results_ks/seed_{seed}/checkpoint.pt" > results_v2/campaign_ks_pre.log 2>&1
echo "ks_pre done $(date)"
python3 run_campaign_v2.py --system ks --label ks_one_step --models "results_rollout_retry_final/ks/seed_{seed}/best_one_step.pt" > results_v2/campaign_ks_one.log 2>&1
python3 run_campaign_v2.py --system ks --label ks_rollout --models "results_rollout_retry_final/ks/seed_{seed}/best_rollout.pt" > results_v2/campaign_ks_roll.log 2>&1
echo "ks arms done $(date)"
python3 run_intervention_eval.py --system ks --root results_rollout_retry_final/ks --label ks_retry > results_v2/intervention_eval_ks.log 2>&1
python3 run_fidelity.py --system ks --models "results_ks/seed_{seed}/checkpoint.pt" --label ks_pre > results_v2/fidelity_ks_pre.log 2>&1
echo "QUEUE_KS_DONE $(date)"
