# ML Rollout Error Amplification

**Code, trained checkpoints, and saved experimental results for _From One-Step Errors to Long-Horizon Rollouts: Realized Error Amplification in Learned Dynamics_.**

This repository investigates a fundamental question in learned simulators and autoregressive dynamical models:

> **How do small one-step prediction errors become large errors over long forecast horizons?**

A learned dynamical system can predict the next state accurately while producing substantially less accurate predictions when its outputs are recursively fed back as inputs. The central premise of this work is that long-horizon error is not determined by one-step accuracy or worst-case sensitivity alone. It also depends on **where model errors are injected, how the dynamics amplify their directions over time, and how propagated errors combine or cancel**.

We analyze these effects through finite-time Jacobian propagators and a first-order forced error equation. We investigate two nonlinear systems and two surrogate architectures:

- **Lorenz–96 (L96):** a chaotic system approximated by a residual multilayer perceptron (MLP).
- **Kuramoto–Sivashinsky (KS):** a nonlinear PDE approximated by a Fourier neural operator (FNO).

The experiments study worst-case transient growth, realized error amplification, defect alignment, vector cancellation, effective spectral dimensionality, and finite-amplitude perturbations. A separate intervention study evaluates whether direct multistep rollout fine-tuning improves forecasting relative to one-step fine-tuning.

**Paper:** [Preprint PDF](https://drive.google.com/file/d/1b3_FYDdXUwYl3ZS6uXUAl1zwuxYyLiSb/view?usp=sharing)

## Repository contents

```text
ml-rollout-error-amplification/
├── experiments_campaign_spectral_jmlr.py   # Corrected L96 / residual MLP diagnostics
├── ks_fno_campaign.py                       # KS / FNO training and diagnostics
├── experiments_campaign_spectral.py        # Original L96 finite-amplitude / PGD study
├── intervention_campaign.py                 # Source for final focused rollout intervention
├── intervention_campaign_v3.py              # Historical four-arm intervention source
│
├── run_primary_campaign_jmlr.ps1            # Five-seed corrected L96 campaign
├── run_ks_fno_campaign.ps1                  # Five-seed KS / FNO campaign
├── run_primary_campaign.ps1                 # Five-seed original PGD campaign
├── run_primary_campaign.sh                  # Bash launcher for original campaign
├── run_rollout_retry_as_recorded.ps1        # Final intervention, recorded hyperparameters
├── run_rollout_retry_final_5seeds.ps1       # Parameterized final intervention launcher
├── run_intervention_final_v3_5seeds.ps1     # Historical intervention launcher
│
├── generate_all_figures.m                   # Single consolidated MATLAB plotting program
├── prepare_results.py                       # Restore losslessly compressed spectral CSVs
├── summarize_final_rollout.py               # Recalculate final intervention table
├── aggregate_interventions.py               # Summaries across independently seeded models
├── cross_system_summary.py                  # Cross-system diagnostic summaries
├── verify_repository.py                     # Check saved result layout and configurations
│
├── results_l96_jmlr/                        # Corrected L96 results and checkpoints, seeds 0–4
├── results_ks/                              # KS / FNO results and checkpoints, seeds 0–4
├── results/                                 # Original L96 PGD/stress results, seeds 0–4
├── results_intervention_final_v3/           # Earlier four-arm results for KS and L96
├── results_rollout_retry_final/             # Final two-arm results for KS and L96
├── tables/                                  # Independent tabular verification outputs
├── tests/                                   # Lightweight Python regression/smoke tests
│
├── requirements.txt
├── pyproject.toml
├── ENVIRONMENT.md
├── REPRODUCIBILITY.md
├── ASSEMBLY.md
├── SOURCE_PROVENANCE.json
├── CITATION.cff
└── LICENSE
```

### Which script produced which results?

| Experiment | Scientific source | Saved result directory |
|---|---|---|
| L96 / residual MLP, reference-trajectory error decomposition | `experiments_campaign_spectral_jmlr.py` | `results_l96_jmlr/seed_0` through `seed_4` |
| KS / FNO, reference-trajectory error decomposition | `ks_fno_campaign.py` | `results_ks/seed_0` through `seed_4` |
| L96 model-trajectory finite-amplitude perturbations and PGD | `experiments_campaign_spectral.py` | `results/seed_0` through `seed_4` |
| Earlier four-arm intervention study | `intervention_campaign_v3.py` | `results_intervention_final_v3/{ks,l96}/seed_0` through `seed_4` |
| **Final focused one-step versus rollout intervention** | **`intervention_campaign.py`** | **`results_rollout_retry_final/{ks,l96}/seed_0` through `seed_4`** |
| Figures and final intervention table | `generate_all_figures.m` | `figures/` (generated locally) |

The similarly named `results/` and `results_l96_jmlr/` directories are **different experiments**: the former stores the original nonlinear stress tests, while the latter stores the corrected reference-trajectory diagnostic campaign. They should not be interchanged when reproducing a figure.

## Quick start: reproduce figures without retraining

**If you only want to regenerate the figures from the supplied data, this is the workflow to use.** It does not train a network, update a checkpoint, or rerun the five-seed campaigns.

From the repository root, install Python dependencies (if necessary) and check the archive:

```powershell
python -m pip install -r requirements.txt
python .\verify_repository.py
```

Restore the spectral-mode CSV files, which are stored in losslessly compressed form to reduce repository size:

```powershell
python .\prepare_results.py
```

Then open MATLAB and execute the **single** consolidated plotting program with the repository directory on the MATLAB path:

```matlab
generate_all_figures
```

The program reads the archived L96, KS, original stress-test, and final intervention outputs, and writes vector PDF plots into `figures/`. It is configured to produce **Figures 2–11**. Figure 1 is a conceptual LaTeX/TikZ schematic rather than a MATLAB figure.

The final rollout-intervention plot and exported table are written as:

```text
figures/fig11_rollout_intervention.pdf
figures/fig11_rollout_intervention_summary.csv
figures/tab_intervention_rollout.tex
```

MATLAB with the **Statistics and Machine Learning Toolbox** is required for the supplied plotting code. The merged MATLAB source was checked structurally, but MATLAB was unavailable in the repository-packaging environment; the rendered figures have not been independently visually checked there.

You can independently recompute the **final intervention summary from the saved JSON files**, without MATLAB and without training:

```powershell
python .\summarize_final_rollout.py
```

This writes `tables/fig11_rollout_intervention_summary.csv`. To run the Python smoke tests:

```powershell
python -m pytest -q
```

## Environment and installation

The recorded Windows research environment included:

- **Python 3.13.14**
- **PyTorch 2.14.0+cpu**
- A **CPU-only PyTorch build**
- **NumPy** (version 2.5.3 appears in KS metadata)
- **SciPy** and **Matplotlib** for supporting analysis
- **MATLAB** with Statistics and Machine Learning Toolbox to regenerate the plotted figures

The requirements file lists required Python packages without claiming an independently verified, cross-platform exact-version lockfile. See [`ENVIRONMENT.md`](ENVIRONMENT.md) for recorded details.

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/Thalesclf/ml-rollout-error-amplification.git
cd ml-rollout-error-amplification
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Note:** The scientific launchers used for the later L96, KS, and intervention campaigns are supplied as Windows PowerShell scripts. The original stress-test pipeline additionally includes a Bash launcher. Running these launchers is separate from regenerating figures and may create or replace result files.

## Experimental systems and saved data

### 1. Lorenz–96: residual MLP

The corrected reference-trajectory study uses the chaotic **Lorenz–96** system, with state dimension 20 and forcing 8, and a residual MLP surrogate. It evaluates independently seeded networks and forecast horizons

`1, 2, 5, 10, 20, 40, 80, 160`.

The main saved outputs, under each `results_l96_jmlr/seed_<seed>/` directory, include:

```text
checkpoint.pt
metadata.json
diagnostic_campaign.csv
defect_terms_campaign.csv
spectral_modes_campaign.csv.gz
```

These tables support the analysis of local defects, finite-time propagator gains, reconstructed errors, alignment, cancellation, and singular-spectrum contributions.

### 2. Kuramoto–Sivashinsky: Fourier neural operator

The second benchmark tests whether the diagnostic framework extends beyond a low-dimensional chaotic system to a nonlinear PDE surrogate. The supplied KS experiment uses a one-dimensional periodic grid with 64 points and an FNO, evaluated at horizons

`1, 2, 4, 8, 16, 32, 64`.

The corresponding data and checkpoints are in `results_ks/seed_<seed>/`, using the same diagnostic, defect-term, and spectral-mode table convention as L96.

### 3. Original finite-amplitude perturbation / PGD study

The original Lorenz–96 campaign is retained separately in `results/`. It contains learned-model checkpoints and stress-test outputs, including:

```text
diagnostic_campaign.csv
defect_terms_campaign.csv
spectral_modes_campaign.csv.gz
perturbation_campaign.csv
epsilon_campaign.csv
epsilon_sweep.csv
checkpoint.pt
```

This study compares directions suggested by finite-time singular-value analysis with nonlinear perturbation growth and projected-gradient-descent (PGD) directions. Its saved perturbation and epsilon tables feed **Figure 10** in the consolidated MATLAB program.

### 4. Intervention experiments

The repository archives two different controlled fine-tuning studies:

- **Historical four-arm comparison:** `one_step`, `local_jacobian`, `rollout`, and `defect_energy`, saved under `results_intervention_final_v3/`.
- **Final focused comparison:** `one_step` versus direct multistep `rollout` fine-tuning, saved under `results_rollout_retry_final/`. These are the data used for the final intervention figure and table.

All intervention arms start from previously trained checkpoints; the intervention campaigns are **fine-tuning studies**, not training-from-scratch comparisons. Each system has five independent pretrained seeds. The results include per-seed configurations, training histories, evaluation summaries, and selected checkpoints.

For the **final focused experiment**, these parameters are taken from the archived `config.json` files, not reverse-engineered from test outcomes:

| Parameter | KS / FNO | L96 / MLP |
|---|---|---|
| Model seeds | 0–4 | 0–4 |
| Compared methods | `one_step,rollout` | `one_step,rollout` |
| Training rollout horizon | 16 | 16 |
| Rollout penalty weight | 0.03 | 0.3 |
| Gradient updates | 60 | 60 |
| Learning rate | `3e-5` | `3e-5` |
| Batch size | 2 | 8 |
| Checkpoint-selection horizons | 16, 32 | 16, 32 |
| Validation trajectory seed | `20261018` | `20261018` |
| Test trajectory seed | `20261019` | `20261019` |

The final `intervention_campaign.py` also contains definitions for other experimental methods; **their presence in the method registry does not mean they were run in this final two-arm comparison**. The earlier implementation is preserved separately as `intervention_campaign_v3.py` because it underlies the historical four-arm output.

## Mathematical framework

Let the reference discrete-time dynamics be

$$
x_{k+1}=F(x_k),
$$

and let the trained surrogate be $f_\theta$. An autonomous model rollout obeys

$$
\hat{x}_{k+1}=f_\theta(\hat{x}_k).
$$

The **one-step defect evaluated along the reference trajectory** is

$$
e_k=f_\theta(x_k)-F(x_k)=f_\theta(x_k)-x_{k+1}.
$$

The Jacobian of the learned dynamics along the reference trajectory is

$$
J_k^{\mathrm{ref}}=Df_\theta(x_k).
$$

For $s_2>s_1$, define the ordered finite-time propagator

$$
\Phi^{\mathrm{ref}}(s_2,s_1)
=J_{s_2-1}^{\mathrm{ref}}J_{s_2-2}^{\mathrm{ref}}\cdots J_{s_1}^{\mathrm{ref}}.
$$

Its largest singular value measures the greatest linear amplification of an infinitesimal perturbation over that interval. The corresponding **worst-case squared gain** is

$$
G_{\max}^{\mathrm{ref}}(t,T)
=\sigma_1^2\!\left(\Phi^{\mathrm{ref}}(t+T,t)\right).
$$

However, an unperturbed forecast initialized at the correct state does not receive only a single initial disturbance. It generates model defects at **every step**. Writing $\delta_k=\hat{x}_k-x_k$ and linearizing around the reference trajectory gives

$$
\delta_{k+1}\approx J_k^{\mathrm{ref}}\delta_k+e_k.
$$

For matching initial conditions, the **first-order forced-error reconstruction** is

$$
\boxed{\displaystyle
\delta_{t+T}^{\mathrm{lin}}
=\sum_{j=0}^{T-1}
\Phi^{\mathrm{ref}}(t+T,t+j+1)\,e_{t+j}
}.
$$

This representation separates three mechanisms:

1. **Defect injection:** how large the error $e_k$ is at each reference state.
2. **Direction-dependent propagation:** how strongly each defect is amplified by its own tail propagator.
3. **Vector composition:** whether the propagated defects reinforce one another or cancel at the final time.

For each injection time, let

$$
q_j=\Phi^{\mathrm{ref}}(t+T,t+j+1)e_{t+j}.
$$

Two useful diagnostics are the **gain efficiency**

$$
\eta_j=\frac{\|q_j\|_2}{\sigma_1\!\left(\Phi^{\mathrm{ref}}(t+T,t+j+1)\right)\,\|e_{t+j}\|_2},
$$

and the **cancellation ratio**

$$
C_T=\frac{\left\|\sum_j q_j\right\|_2}{\sum_j\|q_j\|_2}.
$$

When the denominators are nonzero, $0\leq\eta_j\leq1$ and $0\leq C_T\leq1$. Lower $C_T$ indicates stronger cancellation among contributions. The experiments also use the **full singular spectrum** of tail propagators to quantify how many singular directions account for most of the realized propagated-defect energy.

The key distinction is therefore between **amplification the dynamics could produce** (worst-case gain) and **amplification actually experienced by the model's injected errors** (realized error propagation). Reference-trajectory linearization is a first-order approximation; it may lose accuracy when nonlinear trajectory separation becomes large.

For finite-amplitude sensitivity tests, the analysis also considers Jacobian propagators along the **autonomous learned trajectory**, which are distinct from the reference-trajectory propagators used in the forced-error reconstruction.

## Reproducing the numerical campaigns (optional and computationally expensive)

**You do not need this section to make the plots or inspect the saved results.** These commands launch new model training and/or experimental evaluations. Back up archived data and checkpoints before attempting a rerun.

On Windows PowerShell, the recorded campaign launchers are:

```powershell
.\run_primary_campaign_jmlr.ps1  # Corrected five-seed L96 / MLP diagnostics
.\run_ks_fno_campaign.ps1        # Five-seed KS / FNO diagnostics
.\run_primary_campaign.ps1       # Original five-seed L96 PGD stress test
```

The original PGD campaign also has a Bash launcher:

```bash
bash run_primary_campaign.sh
```

For the **final focused intervention**, the recorded-configuration wrapper is:

```powershell
.\run_rollout_retry_as_recorded.ps1
```

This invokes the parameterized final script with the documented KS and L96 rollout weights and horizons. **It starts fine-tuning again and may overwrite existing saved outputs**; it is not part of the figure-generation procedure.

The source for the earlier historical intervention is also retained, together with `run_intervention_final_v3_5seeds.ps1`. Consult [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for provenance, limitations, and the mapping from scripts to plots.

## Tests and reproducibility checks

Run the automated Python tests and validate the saved result layout:

```powershell
python -m pytest -q
python .\verify_repository.py
```

To summarize the final rollout intervention from the archived evaluations:

```powershell
python .\summarize_final_rollout.py
```

These operations **do not retrain the neural networks**. They check code behavior, repository structure, saved configurations, or results-derived statistics. They should not be confused with independent reproduction of the original five-seed simulations.

The repository includes [`SOURCE_PROVENANCE.json`](SOURCE_PROVENANCE.json), which records source provenance and SHA-256 checksums for retained files, and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), which distinguishes archived outputs from computations carried out during packaging. File checksums document integrity; they do not by themselves establish the exact historical executable state that produced every archived number.

## Data organization and GitHub distribution

All five seed directories are included for the core L96, KS, and original PGD studies, as well as the KS/L96 intervention experiments. The main diagnostic tables are stored in CSV format; intervention configurations and evaluation summaries are JSON; selected trained models are PyTorch `.pt` checkpoints.

The large spectral tables are stored as `spectral_modes_campaign.csv.gz` to keep tracked files manageable. Use `python prepare_results.py` to restore `.csv` copies before MATLAB plotting; these decompressed copies need not be committed.

If you received the repository as multiple ZIP archives, extract the core archive and **all six supplemental data archives into the same parent directory** so they merge into one `ml-rollout-error-amplification/` folder. See [`ASSEMBLY.md`](ASSEMBLY.md). A complete release archive can instead be extracted directly.

For GitHub, review the changes in a cloned working tree before pushing. GitHub enforces a 100 MB limit per ordinary tracked file, and repositories containing hundreds of megabytes of result history may be better distributed using Git LFS or GitHub Release assets. The saved data should remain clearly associated with the source code and manuscript version.

## Citation

The manuscript is currently available through the [preprint link retained from the original README](https://drive.google.com/file/d/1b3_FYDdXUwYl3ZS6uXUAl1zwuxYyLiSb/view?usp=sharing).

Please cite the manuscript if you use the theoretical framework, experimental methods, or archived results. See [`CITATION.cff`](CITATION.cff) for repository citation metadata. This README does not assume a journal acceptance date or invent an arXiv identifier.

## License

This repository is distributed under the **MIT License**. See [`LICENSE`](LICENSE).
