from pathlib import Path
import json

def test_final_rollout_recorded_settings():
    root=Path(__file__).resolve().parents[1]
    base=root/'results_rollout_retry_final'
    if not base.is_dir():
        import pytest
        pytest.skip('Optional archived results not included in source-only package')
    for system, lam in [('ks','rollout=0.03'),('l96','rollout=0.3')]:
        for seed in range(5):
            p=base/system/f'seed_{seed}'
            cfg=json.loads((p/'config.json').read_text())
            assert cfg['horizon']==16
            assert cfg['methods']=='one_step,rollout'
            assert cfg['lambda_by_method']==lam
            assert cfg['test_data_seed']==20261019
            assert {x['method'] for x in json.loads((p/'results.json').read_text())}=={'one_step','rollout'}
