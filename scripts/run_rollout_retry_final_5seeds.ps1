param(
    [Parameter(Mandatory=$true)]
    [int]$KsH,

    [Parameter(Mandatory=$true)]
    [double]$KsLambda,

    [Parameter(Mandatory=$true)]
    [int]$L96H,

    [Parameter(Mandatory=$true)]
    [double]$L96Lambda
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Missing virtualenv Python: $python" }

# Freeze these only after the validation sweep has selected H/lambda.
$valSeed = 20261018
$testSeed = 20261019

$culture = [System.Globalization.CultureInfo]::InvariantCulture
$outRoot = '.\results_rollout_retry_final'

foreach ($System in @('ks','l96')) {
    if ($System -eq 'ks') {
        $H = $KsH
        $lambda = $KsLambda
        $batch = 2
        $ckptRoot = '.\results_ks'
    }
    else {
        $H = $L96H
        $lambda = $L96Lambda
        $batch = 8
        $ckptRoot = '.\results_l96_jmlr'
    }

    $lstr = $lambda.ToString($culture)
    Write-Host ""
    Write-Host "===== FINAL ${System}: H=$H lambda=$lstr ====="

    foreach ($seed in @(0,1,2,3,4)) {
        $checkpoint = "$ckptRoot\seed_$seed\checkpoint.pt"
        if (-not (Test-Path $checkpoint)) { throw "Checkpoint missing: $checkpoint" }

        & $python -u .\scripts/intervention_campaign.py `
            --system $System `
            --seed $seed `
            --checkpoint $checkpoint `
            --outdir "$outRoot\$System\seed_$seed" `
            --device cpu `
            --methods 'one_step,rollout' `
            --steps 60 `
            --batch-size $batch `
            --horizon $H `
            --lr 3e-5 `
            --lambda-by-method "rollout=$lstr" `
            --eval-every 15 `
            --threads 4 `
            --eval-horizons '1,4,8,16,32,64' `
            --selection-horizons '16,32' `
            --val-data-seed $valSeed `
            --test-data-seed $testSeed

        if ($LASTEXITCODE -ne 0) {
            throw "Final run failed: system=$System seed=$seed"
        }
    }
}

Write-Host ""
Write-Host "All final rollout runs completed."
Write-Host "Aggregate with:"
Write-Host "  python .\scripts/aggregate_interventions.py .\results_rollout_retry_final"
