# Reproduces the *recorded configuration* of the final two-arm rollout study.
# These values were read from all ten saved config.json files; not retuned.
# CAUTION: this starts a new training campaign and can replace saved outputs.
# Move or back up results_rollout_retry_final before running it again.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& '.\run_rollout_retry_final_5seeds.ps1' `
    -KsH 16 -KsLambda 0.03 `
    -L96H 16 -L96Lambda 0.3
if ($LASTEXITCODE -ne 0) { throw 'Final rollout rerun failed' }
