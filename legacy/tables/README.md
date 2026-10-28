# Figure 11 data of the first version of the study

`fig11_rollout_intervention_summary.csv` is written by `legacy/summarize_final_rollout.py`
from the 10 `results_rollout_retry_final/{ks,l96}/seed_*/results.json` files of the
historical data (see `docs/DATA_RELEASE.md`). It gives the medians over seeds of the
test relative RMSE of the one-step and rollout arms and of their paired percentage
change, as plotted by the MATLAB script of the first version
(`legacy/matlab/generate_all_figures_historical.m`), and `WinCount`, the number of
seeds in which the rollout arm has the lower error. It is kept for reference and is
not used by the current manuscript.
