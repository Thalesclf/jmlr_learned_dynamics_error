#!/bin/bash
# core 1: after the L96 extended queue, the rollout arm of the KS fine-tuning study
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Written for the layout used until October 2026 (scripts and results_v2/ at the top level of the project).
cd "$(dirname "$0")/.."
until grep -q QUEUE_EXT_DONE results_v2/queue_extended.log 2>/dev/null; do sleep 20; done
python3 run_campaign_v2.py --system ks --label ks_rollout --models "results_rollout_retry_final/ks/seed_{seed}/best_rollout.pt" > results_v2/campaign_ks_roll.log 2>&1
echo "QUEUE_KS_C_DONE $(date)"
