#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Written for the layout used until October 2026 (scripts and results_v2/ at the top level of the project).
cd "$(dirname "$0")/.."
until grep -q QUEUE_L96_DONE results_v2/queue_l96.log 2>/dev/null; do sleep 30; done
python3 run_campaign_v2.py --system ks --label ks_pre --models "results_ks/seed_{seed}/checkpoint.pt" > results_v2/campaign_ks_pre.log 2>&1
python3 run_fidelity.py --system ks --models "results_ks/seed_{seed}/checkpoint.pt" --label ks_pre > results_v2/fidelity_ks_pre.log 2>&1
echo "QUEUE_KS_PRE_DONE $(date)"
