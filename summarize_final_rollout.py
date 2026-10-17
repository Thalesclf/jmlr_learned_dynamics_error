"""Compute Figure 11 medians directly from saved final intervention JSON files.

Use: python summarize_final_rollout.py
Writes tables/fig11_rollout_intervention_summary.csv, with additional WinCount.
No model retraining or inspection of other experiment result directories.
"""
from pathlib import Path
import argparse, json, csv, statistics

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'results_rollout_retry_final')
p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'tables/fig11_rollout_intervention_summary.csv')
a=p.parse_args()

rows=[]
for system in ('l96','ks'):
    pair=[]
    for seed in range(5):
        src=a.root/system/f'seed_{seed}'/'results.json'
        d={entry['method']:entry['test'] for entry in json.loads(src.read_text(encoding='utf-8'))}
        if set(d)!={'one_step','rollout'}:
            raise ValueError(f'Unexpected methods in {src}: {list(d)}')
        pair.append(d)
    for h in (1,4,8,16,32,64):
        key=f'h{h}_rel_rmse'
        one=[x['one_step'][key] for x in pair]
        rollout=[x['rollout'][key] for x in pair]
        rows.append({'System':system.upper(),'Horizon':h,
                     'OneStepRelRMSE':statistics.median(one),
                     'RolloutRelRMSE':statistics.median(rollout),
                     'PairedPercentChange':statistics.median([100*(r/o-1) for o,r in zip(one,rollout)]),
                     'WinCount':sum(r<o for o,r in zip(one,rollout))})
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w',newline='',encoding='utf-8') as file:
    writer=csv.DictWriter(file,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
print(f'Saved {a.output}')
for row in rows:
    if row['Horizon']==64:
        print(row)
