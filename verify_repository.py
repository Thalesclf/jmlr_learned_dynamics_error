"""Validate the archived final JMLR research outputs (no numerical retraining)."""
from pathlib import Path
import json, gzip, hashlib, sys
root=Path(__file__).resolve().parent
issues=[]
for ds in ('results','results_l96_jmlr','results_ks'):
    for seed in range(5):
        sub=root/ds/f'seed_{seed}'
        required=['checkpoint.pt','diagnostic_campaign.csv','defect_terms_campaign.csv','spectral_modes_campaign.csv.gz']
        if ds=='results':required+=['perturbation_campaign.csv','epsilon_campaign.csv']
        for name in required:
            p=sub/name
            if not p.is_file(): issues.append(f'MISSING: {p.relative_to(root)}')
for ds in ('results_intervention_final_v3','results_rollout_retry_final'):
    for sysname in ('ks','l96'):
        for seed in range(5):
            sub=root/ds/sysname/f'seed_{seed}'
            required=['config.json','results.json']
            if ds=='results_rollout_retry_final':
                required+=['training_history.csv','best_one_step.pt','best_rollout.pt']
            for name in required:
                p=sub/name
                if not p.is_file():issues.append(f'MISSING: {p.relative_to(root)}')
            if not (sub/'config.json').is_file() or not (sub/'results.json').is_file():continue
            cfg=json.loads((sub/'config.json').read_text())
            result=json.loads((sub/'results.json').read_text())
            if cfg.get('system')!=sysname or cfg.get('seed')!=seed:
                issues.append(f'Wrong system/seed: {sub.relative_to(root)}')
            if ds=='results_rollout_retry_final':
                expected='rollout=0.03' if sysname=='ks' else 'rollout=0.3'
                checks={'horizon':16,'steps':60,'selection_horizons':'16,32',
                        'methods':'one_step,rollout','lambda_by_method':expected,
                        'val_data_seed':20261018,'test_data_seed':20261019}
                for k,v in checks.items():
                    if cfg.get(k)!=v:issues.append(f'Unexpected {k} in {sub.relative_to(root)}: {cfg.get(k)!r} != {v!r}')
                if {i.get('method') for i in result}!={'one_step','rollout'}:
                    issues.append(f'Unexpected methods in {sub.relative_to(root)}')
                for entry in result:
                    if any(f'h{h}_rel_rmse' not in entry.get('test',{}) for h in (1,4,8,16,32,64)):
                        issues.append(f'Missing horizon metrics in {sub.relative_to(root)}')
print('Primary L96, KS, and stress-test data:', 'complete' if not issues else 'see errors')
print('Final rollout: all 10 seed configurations and two methods checked')
if issues:
    for issue in issues:print(issue,file=sys.stderr)
    raise SystemExit(1)
print('ARCHIVED DATA VALIDATION PASSED (not an independent training reproduction)')
