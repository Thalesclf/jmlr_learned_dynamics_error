#!/bin/bash
# Extended fine-tuning budget for the converged L96-2000 surrogates (600 updates, lr 1e-4).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Written for the layout used until October 2026 (scripts and results_v2/ at the top level of the project).
cd "$(dirname "$0")/.."
until grep -q QUEUE_L96_DONE results_v2/queue_l96.log 2>/dev/null; do sleep 30; done
python3 run_intervention_np.py --models "results_v2/l96_models/e2000/seed_{seed}/model.npz" --label l96_e2000_ext \
   --steps 600 --lr 1e-4 --eval-every 50 --H 8,16,32 --lam 0.1,0.3,1,3 --validation-only > results_v2/intervention_np_ext_val.log 2>&1
python3 run_intervention_np.py --models "results_v2/l96_models/e2000/seed_{seed}/model.npz" --label l96_e2000_ext \
   --steps 600 --lr 1e-4 --eval-every 50 > results_v2/intervention_np_ext_final.log 2>&1
python3 run_intervention_eval.py --system l96 --root results_v2/intervention_np/l96_e2000_ext --label l96_e2000_ext --ext npz \
   --initial "results_v2/l96_models/e2000/seed_{seed}/model.npz" > results_v2/intervention_eval_ext.log 2>&1
python3 run_campaign_v2.py --system l96 --label l96_e2000_ext_one --models "results_v2/intervention_np/l96_e2000_ext/seed_{seed}/best_one_step.npz" > results_v2/campaign_ext_one.log 2>&1
python3 run_campaign_v2.py --system l96 --label l96_e2000_ext_roll --models "results_v2/intervention_np/l96_e2000_ext/seed_{seed}/best_rollout.npz" > results_v2/campaign_ext_roll.log 2>&1
python3 run_prospective.py --system l96 --models "results_v2/l96_models/e2000/seed_{seed}/model.npz" --label l96_e2000 --campaign l96_e2000 > results_v2/prospective_l96_e2000.log 2>&1
echo "QUEUE_EXT_DONE $(date)"
