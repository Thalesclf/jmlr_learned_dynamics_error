# Historical data

The original PyTorch checkpoints and the results of the campaigns run with them are not part of the Git repository. They are distributed as two archives attached to release v1.0.0:

https://github.com/Thalesclf/jmlr_learned_dynamics_error/releases/tag/v1.0.0

| Archive | Size (bytes) | SHA-256 |
|---|---|---|
| `Historical_reproduction_data_part1.zip` | 169,971,148 | `075fbac608459aed0e15d49cbc8fe82772cb04aca807610e7643357b56b286e6` |
| `Historical_reproduction_data_part2.zip` | 176,824,872 | `91299d845377c02b5534f230b89068134f7e2836345caec997cf17a17fe2b4fe` |

The tables and figures of the manuscript are generated from `results/current/` and do not need these archives. They are needed to recompute the diagnostics of the original checkpoints and by the four tests that check the historical files. For example, `python scripts/run_campaign_v2.py --system l96 --label l96_e20 --models "results/seed_{seed}/checkpoint.pt" --outroot rerun/campaign` recomputes the L96-20 campaign in `rerun/campaign/l96_e20/`, which can be compared with `results/current/campaign/l96_e20/`.

## Contents

The two archives hold 225 files, among them 75 checkpoints, in the folders used by the original scripts.

| Folder | Archive | Contents |
|---|---|---|
| `results/seed_0` to `seed_4` | part 2 | L96-20 surrogates (`checkpoint.pt`) and the diagnostic and finite-amplitude perturbation campaigns of the first version of the study |
| `results_l96_jmlr/seed_*` | part 1 | the same L96-20 weights, saved by a later run of the diagnostic campaign and used as starting points of the fine-tuning study |
| `results_ks/seed_*` | part 1 | Fourier neural operators for Kuramoto-Sivashinsky and their diagnostic campaign |
| `results_rollout_retry_final/{l96,ks}/seed_*` | part 2 | fine-tuning study of the manuscript for L96-20 and KS: configuration, training history, test results and the checkpoints selected for the one-step and rollout arms |
| `results_intervention_final_v3/{l96,ks}/seed_*` | part 2 | earlier fine-tuning study with four objectives (one-step, local Jacobian, rollout and defect energy), not used in the manuscript |

Each archive contains `HISTORICAL_RELEASE_MANIFEST.json` with the SHA-256 hash of every file. The uncompressed copies of the 15 `spectral_modes_campaign.csv.gz` files are not included, since they can be recreated from the compressed ones.

## Restoring the data

Download the two archives to a folder outside the repository, check their hashes against the table above and restore them from the repository root:

```bash
# in a folder outside the repository
curl -LO https://github.com/Thalesclf/jmlr_learned_dynamics_error/releases/download/v1.0.0/Historical_reproduction_data_part1.zip
curl -LO https://github.com/Thalesclf/jmlr_learned_dynamics_error/releases/download/v1.0.0/Historical_reproduction_data_part2.zip
sha256sum Historical_reproduction_data_part1.zip Historical_reproduction_data_part2.zip

# from the repository root
python scripts/restore_legacy_data.py /path/to/Historical_reproduction_data_part1.zip
python scripts/restore_legacy_data.py /path/to/Historical_reproduction_data_part2.zip
python scripts/verify_repository.py --legacy
python -m pytest -q
```

`restore_legacy_data.py` writes the files to the paths listed in the manifest, checks the SHA-256 hash of each one and stops if a file with different contents already exists. The restored folders are listed in `.gitignore`. Some legacy scripts read the uncompressed spectral files, which `python scripts/restore_legacy_data.py --materialize-csv` recreates.

With both archives restored and without PyTorch, the test suite gives 24 passed and 3 skipped tests. The three skipped modules test the PyTorch implementation and run when PyTorch is installed.