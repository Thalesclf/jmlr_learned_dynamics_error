#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Written for the layout used until October 2026 (scripts and results_v2/ at the top level of the project).
cd "$(dirname "$0")/.."
until [ -f results_v2/l96_models/e150/seed_4/meta.json ]; do sleep 20; done
echo "models ready $(date)"
rm -rf results_v2/campaign/l96_e20
python3 run_campaign_v2.py --system l96 --label l96_e2000 --models "results_v2/l96_models/e2000/seed_{seed}/model.npz" > results_v2/campaign_l96_e2000.log 2>&1
python3 run_campaign_v2.py --system l96 --label l96_e20 --models "results/seed_{seed}/checkpoint.pt" > results_v2/campaign_l96_e20.log 2>&1
python3 run_campaign_v2.py --system l96 --label l96_e150 --models "results_v2/l96_models/e150/seed_{seed}/model.npz" > results_v2/campaign_l96_e150.log 2>&1
echo "campaigns done $(date)"
python3 run_fidelity.py --system l96 --models "results_v2/l96_models/e2000/seed_{seed}/model.npz" --label l96_e2000 > results_v2/fidelity_l96_e2000.log 2>&1
python3 run_fidelity.py --system l96 --models "results_v2/l96_models/e150/seed_{seed}/model.npz" --label l96_e150 > results_v2/fidelity_l96_e150.log 2>&1
python3 run_fidelity.py --system l96 --models "results/seed_{seed}/checkpoint.pt" --label l96_e20 > results_v2/fidelity_l96_e20.log 2>&1
echo "fidelity done $(date)"
python3 run_intervention_np.py --models "results_v2/l96_models/e2000/seed_{seed}/model.npz" --label l96_e2000 --validation-only > results_v2/intervention_np_e2000_val.log 2>&1
python3 run_intervention_np.py --models "results_v2/l96_models/e2000/seed_{seed}/model.npz" --label l96_e2000 > results_v2/intervention_np_e2000_final.log 2>&1
python3 run_intervention_eval.py --system l96 --root results_v2/intervention_np/l96_e2000 --label l96_e2000_np --ext npz > results_v2/intervention_eval_e2000.log 2>&1
python3 run_campaign_v2.py --system l96 --label l96_e2000_ft_one --models "results_v2/intervention_np/l96_e2000/seed_{seed}/best_one_step.npz" > results_v2/campaign_l96_e2000_ft_one.log 2>&1
python3 run_campaign_v2.py --system l96 --label l96_e2000_ft_roll --models "results_v2/intervention_np/l96_e2000/seed_{seed}/best_rollout.npz" > results_v2/campaign_l96_e2000_ft_roll.log 2>&1
python3 run_campaign_v2.py --system l96 --label l96_e20_ft_one --models "results_rollout_retry_final/l96/seed_{seed}/best_one_step.pt" --norm-from "results/seed_{seed}/checkpoint.pt" > results_v2/campaign_l96_e20_ft_one.log 2>&1
python3 run_campaign_v2.py --system l96 --label l96_e20_ft_roll --models "results_rollout_retry_final/l96/seed_{seed}/best_rollout.pt" --norm-from "results/seed_{seed}/checkpoint.pt" > results_v2/campaign_l96_e20_ft_roll.log 2>&1
echo "intervention done $(date)"
python3 run_pgd_np.py --models "results_v2/l96_models/e2000/seed_{seed}/model.npz" --label l96_e2000 > results_v2/pgd_e2000.log 2>&1
echo "QUEUE_L96_DONE $(date)"
