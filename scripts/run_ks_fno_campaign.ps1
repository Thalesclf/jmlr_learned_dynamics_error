$seeds = 0,1,2,3,4

foreach ($seed in $seeds) {
    python scripts/ks_fno_campaign.py `
        --seed $seed `
        --data-seed 1729 `
        --outdir "results_ks/seed_$seed" `
        --device "cpu" `
        --dtype "float64" `
        --n-grid 64 `
        --length 22 `
        --dt 0.25 `
        --sample-stride 1 `
        --data-steps 12000 `
        --burnin 2000 `
        --modes 16 `
        --width 32 `
        --depth 4 `
        --projection-width 64 `
        --epochs 50 `
        --batch-size 128 `
        --lr 1e-3 `
        --horizons "1,2,4,8,16,32,64" `
        --n-test-states 50 `
        --n-spectral-states 15
}
