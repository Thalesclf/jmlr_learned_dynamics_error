param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('ks','l96')]
    [string]$System
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Missing virtualenv Python: $python" }
if (-not (Test-Path '.\scripts/intervention_campaign.py')) { throw "Missing scripts/intervention_campaign.py" }

# Fresh validation trajectory for this renewed rollout-training study.
$valSeed = 20261018

$horizons = @(4, 8, 16)
$lambdas  = @(0.01, 0.03, 0.1, 0.3)

if ($System -eq 'ks') {
    $checkpoint = '.\results_ks\seed_0\checkpoint.pt'
    $batch = 2
}
else {
    $checkpoint = '.\results_l96_jmlr\seed_0\checkpoint.pt'
    $batch = 8
}

if (-not (Test-Path $checkpoint)) { throw "Checkpoint missing: $checkpoint" }

$culture = [System.Globalization.CultureInfo]::InvariantCulture

foreach ($H in $horizons) {
    foreach ($lambda in $lambdas) {
        $lstr = $lambda.ToString($culture)
        $ltag = $lstr.Replace('.', 'p')
        $outdir = ".\results_rollout_retry\_tuning\$System\H${H}_lambda_${ltag}"

        Write-Host ""
        Write-Host "===== $System | H=$H | lambda=$lstr ====="

        & $python -u .\scripts/intervention_campaign.py `
            --system $System `
            --seed 0 `
            --checkpoint $checkpoint `
            --outdir $outdir `
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
            --validation-only

        if ($LASTEXITCODE -ne 0) {
            throw "Sweep failed: system=$System H=$H lambda=$lstr"
        }
    }
}

Write-Host ""
Write-Host "Validation-only rollout sweep finished for $System."
Write-Host "Summarize with:"
Write-Host "  python .\scripts/summarize_rollout_retry.py .\results_rollout_retry\_tuning\$System"
Write-Host ""
Write-Host "Do NOT run a new test trajectory yet."
