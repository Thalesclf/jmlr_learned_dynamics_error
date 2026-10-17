# Assemble the GitHub-ready research suite

For convenience the package is distributed in the core ZIP and six independent
data ZIPs. **No split-volume decompression program is needed**; each archive
contains paths relative to the same `ml-rollout-error-amplification/` folder.

1. Extract `ml_rollout_final_core_github.zip` to a new empty parent folder.
2. Extract `ml_rollout_final_l96_seeds_0-1.zip`, `ml_rollout_final_l96_seeds_2-3.zip`,
   `ml_rollout_final_l96_seed_4.zip`, `ml_rollout_final_stress_seeds_0-1.zip`,
   `ml_rollout_final_stress_seeds_2-3.zip`, and `ml_rollout_final_stress_seed_4.zip`
   to **the same parent folder**.
3. Run `python verify_repository.py` inside `ml-rollout-error-amplification/`.
4. Run `python prepare_results.py` before running `generate_all_figures` in MATLAB.

If uploading *code only* to GitHub, use `ml_rollout_final_code_only.zip` instead.
It lacks checkpoints/results and cannot regenerate figures until data are added.

To assemble in Windows PowerShell, with these ZIPs in the current directory:

```powershell
Expand-Archive .\ml_rollout_final_core_github.zip -DestinationPath . -Force
Get-ChildItem .\ml_rollout_final_l96_*.zip, .\ml_rollout_final_stress_*.zip |
  ForEach-Object { Expand-Archive $_.FullName -DestinationPath . -Force }
cd .\ml-rollout-error-amplification
python .\verify_repository.py
python .\prepare_results.py
```

The full release is also distributed as `ml_rollout_complete_final_github.zip`;
that single archive is larger but needs no assembly.
