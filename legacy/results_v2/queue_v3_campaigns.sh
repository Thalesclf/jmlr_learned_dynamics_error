#!/bin/bash
# Re-run of the main diagnostic campaigns with the unbiased (Monte Carlo norm) efficiency null
# and the iid-defect coherence null (realized_error/diagnostics.py, revision of 24 Oct 2026).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/.."
python3 run_campaign_v2.py --system l96 --label l96_e2000 --models "results_v2/l96_models/e2000/seed_{seed}/model.npz" > results_v2/campaign_l96_e2000.log 2>&1
python3 run_campaign_v2.py --system l96 --label l96_e150 --models "results_v2/l96_models/e150/seed_{seed}/model.npz" > results_v2/campaign_l96_e150.log 2>&1
python3 run_campaign_v2.py --system l96 --label l96_e20 --models "results/seed_{seed}/checkpoint.pt" > results_v2/campaign_l96_e20.log 2>&1
echo "L96_DONE $(date)"
python3 run_campaign_v2.py --system ks --label ks_pre --models "results_ks/seed_{seed}/checkpoint.pt" > results_v2/campaign_ks_pre.log 2>&1
echo "QUEUE_V3_DONE $(date)"
