param(
    [ValidateSet('ks','l96')][string]$System='ks',
    [Parameter(Mandatory=$true)][double]$LambdaLocal,
    [Parameter(Mandatory=$true)][double]$LambdaRollout,
    [Parameter(Mandatory=$true)][double]$LambdaDefectEnergy
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Missing virtualenv Python: $python" }
$finalTestSeed = 424242
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$weights = "local_jacobian=$($LambdaLocal.ToString($culture)),rollout=$($LambdaRollout.ToString($culture)),defect_energy=$($LambdaDefectEnergy.ToString($culture))"
Write-Host "Final v3 experiment: $System, weights: $weights; test seed: $finalTestSeed"
foreach ($seed in @(0,1,2,3,4)) {
    if ($System -eq 'ks') { $checkpoint = "results_ks/seed_$seed/checkpoint.pt"; $batch = 2 }
    else { $checkpoint = "results_l96_jmlr/seed_$seed/checkpoint.pt"; $batch = 8 }
    if (-not (Test-Path $checkpoint)) { throw "Checkpoint missing: $checkpoint" }
    & $python -u legacy/intervention_campaign_v3.py `
        --system $System --seed $seed --checkpoint $checkpoint `
        --outdir "results_intervention_final_v3/$System/seed_$seed" `
        --device cpu --methods 'one_step,local_jacobian,rollout,defect_energy' `
        --steps 60 --batch-size $batch --horizon 16 --lr 3e-5 `
        --lambda-by-method $weights --eval-every 15 --threads 4 `
        --eval-horizons '1,4,8,16,32,64' --selection-horizons '16,32' `
        --test-data-seed $finalTestSeed
    if ($LASTEXITCODE -ne 0) { throw "Experiment stopped at seed $seed; exit $LASTEXITCODE" }
}
Write-Host 'All five seeds completed. Aggregate with: python scripts/aggregate_interventions.py results_intervention_final_v3'
