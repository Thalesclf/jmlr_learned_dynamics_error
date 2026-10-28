# Realized Error Amplification in Learned Dynamical Systems

Code, saved results and LaTeX sources of the manuscript *Realized Error Amplification in Learned Dynamical Systems* by Thales Coelho Leite Fava, submitted to the Journal of Machine Learning Research.

**Paper (PDF):** GOOGLE_DRIVE_LINK

The rollout error of an autoregressive surrogate is written, to first order, as a sum of one-step defects transported by the tangent dynamics. An exact factorization splits it into the defect magnitude, the worst-case gain of the tail propagators, the efficiency with which the defects use that gain, the cancellation among the transported defects and a nonlinear remainder. The study covers residual MLPs trained on Lorenz-96 with three training budgets (L96-20, L96-150 and L96-2000) and Fourier neural operators trained on Kuramoto-Sivashinsky (KS), with five seeds each.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python scripts/verify_repository.py --checksums
```

The tests and the regeneration of all tables and figures were run with Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0 and Matplotlib 3.11. PyTorch is needed only by the original PyTorch training and campaign scripts (`python -m pip install -e ".[torch]"`), since the NumPy code reads the original checkpoints without it. Without PyTorch, the three test modules that use it are skipped, and four more tests are skipped until the historical data are restored (see below).

## Reproducing the tables and figures

All tables and figures of the manuscript are generated from the saved results in `results/current/`:

```bash
bash scripts/00_reproduce_paper.sh
```

or, step by step, from the repository root:

```bash
python scripts/verify_repository.py --checksums                  # check the saved files first
python scripts/analysis.py l96_e20 l96_e150 l96_e2000 ks_pre    # tables/horizons_*.csv, tables/spectral_*.csv
python scripts/build_tables.py                                   # LaTeX tables and tables/paper_numbers.json
python scripts/plot_figures.py                                   # figures/fig_*.pdf
python scripts/export_matlab_plot_data.py                        # plotted values of every figure (CSV)
python scripts/verify_repository.py
cd paper && latexmk -pdf main_jmlr.tex
```

With Make: `make analysis tables figures figure-data verify paper`. The checksums are checked first, because the PDF files written by Matplotlib differ in their bytes between Matplotlib versions even when the figures are the same. [`docs/PAPER_OUTPUTS.md`](docs/PAPER_OUTPUTS.md) lists the script and the results behind every figure and table.

The same figures can be drawn in MATLAB (R2021a or newer) by running `scripts/generate_all_figures.m` from the repository root. The script reads the plotted values from `data/processed/figure_data/` and saves the figures in `figures/matlab/`.

The manuscript sources in `paper/` read the figures and tables from `../figures` and `../tables`. `python scripts/export_overleaf.py Overleaf_JMLR.zip` writes a standalone copy for Overleaf, and `python scripts/sync_overleaf.py <overleaf-folder>` copies regenerated figures and tables into an existing copy.

## Historical data

The original PyTorch checkpoints (L96-20 surrogates, FNOs and their fine-tuned versions) and the results of the campaigns run with them are distributed as two archives attached to release [v1.0.0](https://github.com/Thalesclf/jmlr_learned_dynamics_error/releases/tag/v1.0.0). [`docs/DATA_RELEASE.md`](docs/DATA_RELEASE.md) gives their SHA-256 hashes and the commands that restore them. The tables and figures do not need these archives, which are needed only to recompute the diagnostics of the original checkpoints.

## Repository layout

| Path | Contents |
|---|---|
| `src/realized_error/` | NumPy implementation of the reference systems and their tangent maps, the two surrogates, training and all diagnostics, and a reader for PyTorch checkpoints that does not need PyTorch |
| `scripts/` | training, diagnostic campaigns, fine-tuning study, analysis, tables, figures, verification and restoration of the historical data |
| `results/current/` | saved results of all campaigns used in the manuscript and the Lorenz-96 surrogates trained with NumPy |
| `tables/`, `figures/` | generated tables (LaTeX and CSV) and figures (PDF) |
| `data/processed/figure_data/` | plotted values of every figure (CSV), read by `scripts/generate_all_figures.m` |
| `paper/` | LaTeX sources of the manuscript |
| `tests/` | tests of the numerical routines and of the saved results |
| `docs/` | historical data, provenance of the figures and tables, reproducibility notes |
| `legacy/` | scripts and records of earlier versions of the study, kept for reference |

## Limitations

- Regenerating the tables and figures from the saved results does not repeat the experiments. The training campaigns were not rerun for this release, and a new training run depends on the PyTorch version, the number of threads and the hardware. The original PyTorch models were trained with PyTorch 2.14.0 (CPU).
- `scripts/generate_all_figures.m` has not been run in MATLAB for this release, and the figures of the manuscript are those written by `scripts/plot_figures.py`.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) gives the details.

## Citation and license

If you use the code or the results, please cite the manuscript:

> Thales Coelho Leite Fava. Realized error amplification in learned dynamical systems. Manuscript submitted to the Journal of Machine Learning Research, 2026. GOOGLE_DRIVE_LINK

```bibtex
@unpublished{fava2026realized,
  author = {Fava, Thales Coelho Leite},
  title  = {Realized Error Amplification in Learned Dynamical Systems},
  note   = {Manuscript submitted to the Journal of Machine Learning Research},
  year   = {2026},
  url    = {GOOGLE_DRIVE_LINK}
}
```

The same metadata are in [`CITATION.cff`](CITATION.cff), which GitHub shows under *Cite this repository*. The code is released under the MIT license ([`LICENSE`](LICENSE)).
