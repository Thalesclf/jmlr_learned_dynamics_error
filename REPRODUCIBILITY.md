# Scientific reproducibility and provenance

Input result archives supplied on October 16, 2026:
- `results_l96_jmlr(1).zip`: five corrected L96 seeds
- `results_ks(1).zip`: five KS/FNO seeds
- `results(3).zip`: five original finite-amplitude stress-test seeds
- `results_rollout_retry_final(1).zip`: ten five-seed final intervention runs (KS + L96)

The final focused study's saved configurations confirm rollout H=16,
lambda=0.03 (KS), lambda=0.3 (L96), 60 steps, selection horizons 16 and 32,
validation seed 20261018 and test seed 20261019. A recorded-arguments
wrapper calls the original parameterized PowerShell launcher.

The `intervention_campaign.py` retained here is the exact supplied later
six-arm-capable source (`intervention_campaign(4).py`), whose registry matches
the saved final configuration. The older five-arm source is retained as
`intervention_campaign_v3.py` for historical four-arm results. The algorithm
in either has not been refactored.

All overlapping scientific data archived in this release were byte-for-byte
identical to the previous consolidated repository. The original PGD checkpoint
files and the final rollout campaign are the substantive data additions.
Only lossless gzip compression was used for the large spectral-mode CSVs.

Figures 2--9: `results_l96_jmlr` and `results_ks`.
Figure 10: `results` (model-trajectory stress test).
Figure 11: `results_rollout_retry_final` (final one-step vs rollout study).

Check `tables/fig11_rollout_intervention_summary.csv` for an independent
numerical summary of the final JSONs. The MATLAB figure itself requires
MATLAB; it was not run or visually validated here.

Python tests check implementation and archived layouts but do not constitute
an independently repeated five-seed simulation. Recorded PyTorch and Python
versions may differ from a current environment. No arXiv ID is invented.
