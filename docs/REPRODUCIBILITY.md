# Reproducibility

## From the saved results to the manuscript

`results/current/` contains the outputs of every computation used in the manuscript: the windows and term archives of the diagnostic campaigns (`campaign/`), the ensemble and prospective scores, the fidelity of the surrogates, the evaluation of the fine-tuning study, the stress test, the tangent checks, the Lyapunov exponents of the reference systems and the Lorenz-96 surrogates trained with NumPy (`l96_models/`). `bash scripts/00_reproduce_paper.sh` regenerates all tables and figures from these files and compiles the manuscript.

With Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0 and Matplotlib 3.11, the regenerated LaTeX tables, CSV summaries, `paper_numbers.json`, figures and plotted coordinates are identical, byte for byte, to the files in the repository. With another Matplotlib version, the figure PDFs differ in their bytes and possibly in details of the text rendering.

`SHA256SUMS.txt` lists the SHA-256 hash of every file in `results/current/`, `tables/`, `figures/` and `data/processed/`, and `python scripts/verify_repository.py --checksums` checks them. The check should be run before outputs are regenerated in place. Without options, `verify_repository.py` checks the structure of the saved results: five seeds per campaign, 900 windows per seed, the exactness of the factorization in every window (the five factors multiply to the error within a relative difference of 1e-7), the term archives, the weights of the NumPy surrogates and the layout of the fine-tuning evaluation.

## Tests

`python -m pytest -q` checks the tangent maps and Jacobians against finite differences, the null models against direct Monte Carlo sampling, the exactness of the factorization and of the variance attribution, the saved results and the layout of the repository.

| Setting | Result |
|---|---|
| without PyTorch, without the historical data | 19 passed, 7 skipped |
| without PyTorch, with the historical data restored ([`DATA_RELEASE.md`](DATA_RELEASE.md)) | 23 passed, 3 skipped |

The three modules skipped in the second case (`tests/test_smoke.py`, `tests/test_intervention.py` and `tests/test_ks_fno.py`) need PyTorch and were not run in the environment used to prepare this release.

## Saved metadata

When the results were moved to `results/current/`, the model paths that began with `results_v2/` were changed to `results/current/` in 21 metadata files: the `meta.json` files of the campaigns `l96_e150`, `l96_e2000`, `l96_e2000_ct_one` and `l96_e2000_ct_roll` (five seeds each) and `intervention_np/l96_e2000_ct/protocol.json`. Nothing else was changed. With this substitution reversed, all 235 files in `results/current/` match the hashes in `legacy/SHA256SUMS_original.txt`, the checksum list of the earlier layout, and `legacy/RENAME_LOG.md` gives the earlier path of every file.

## Recomputing the diagnostics

With the historical data restored, the diagnostic campaigns of the original checkpoints can be recomputed with `scripts/run_campaign_v2.py` and the `--outroot` option (see [`DATA_RELEASE.md`](DATA_RELEASE.md)). For seed 0 of the L96-20 surrogates and of the FNOs, the recomputed `windows.csv` files were identical to the saved ones, the term archives had the same contents, and the metadata differed only in the recorded run times (about half a minute for the L96-20 surrogate and five minutes for the FNO).

## Training

The training campaigns were not rerun for this release. The models of the historical data (L96-20 surrogates, FNOs and their fine-tuned versions) were trained with PyTorch 2.14.0 (CPU). This version is recorded in the metadata of the FNOs and of the fine-tuning runs and in the environment listed in `legacy/README_original_archival.md`. The L96-150 and L96-2000 surrogates in `results/current/l96_models/` were trained with NumPy by `scripts/run_l96_models.py`. A new training run is not expected to reproduce the saved models bit for bit, since the result depends on the library versions, the number of threads and the hardware. The Kuramoto-Sivashinsky reference trajectories are reproducible on a given platform but not bit for bit across platforms (Appendix F of the manuscript).

## MATLAB

`scripts/generate_all_figures.m` reads the tables of `data/processed/figure_data/` and draws the nine figures with MATLAB graphics in `figures/matlab/` (MATLAB R2021a or newer, run from the repository root). It has not been run in MATLAB for this release, so its output should be compared with the PDFs in `figures/` before it is used.

## Statistical dependence

The 312 evaluation windows of the fine-tuning study are non-overlapping windows of a single held-out trajectory and are not necessarily independent, so the window-bootstrap intervals are conditional on that trajectory (Section 8 of the manuscript). Within a seed, the windows of the diagnostic campaigns are also correlated, and the variance shares describe the variability across initial states, not a causal decomposition.
