"""Pool campaign CSVs and print the cross-system diagnostic hierarchy.

Usage examples:
  python cross_system_summary.py --l96 results_l96_jmlr --ks results_ks
  python cross_system_summary.py --ks results_ks
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import numpy as np


def read_seed_csvs(root: Path, filename: str):
    rows=[]
    for p in sorted(root.glob(f"seed_*/{filename}")):
        with p.open(newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def f(row,key):
    try: return float(row[key])
    except Exception: return math.nan


def rankdata(a):
    order=np.argsort(a,kind='mergesort'); ranks=np.empty(len(a),float); i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]: j+=1
        ranks[order[i:j]]=0.5*(i+j-1)+1; i=j
    return ranks


def spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); q=np.isfinite(x)&np.isfinite(y); x=x[q]; y=y[q]
    if len(x)<2: return math.nan
    rx=rankdata(x); ry=rankdata(y); rx-=rx.mean(); ry-=ry.mean(); d=np.linalg.norm(rx)*np.linalg.norm(ry)
    return float(np.dot(rx,ry)/d) if d else math.nan


def summarize(label, rows):
    horizons=sorted({int(float(r['horizon'])) for r in rows})
    keys=[
        ('Gmax','Gmax'),
        ('spectral_radius_J','rho(J)'),
        ('sigma1_J','sigma1(J)'),
        ('fro_J','||J||F'),
        ('defect_l2_sum','sum||e||'),
        ('propagated_defect_norm_sum','sum||Phi e||'),
        ('linear_predicted_error','||sum Phi e||'),
        ('defect_amplification_bound','B_T'),
    ]
    print(f"\n## {label}")
    print('| T | ' + ' | '.join(n for _,n in keys) + ' | median Rlin | median C_T |')
    print('|---:' + '|---:'*(len(keys)+2) + '|')
    for T in horizons:
        rr=[r for r in rows if int(float(r['horizon']))==T]
        y=[f(r,'actual_final_error') if 'actual_final_error' in r else f(r,'rollout_error') for r in rr]
        vals=[]
        for k,_ in keys:
            if not rr or k not in rr[0]: vals.append(math.nan)
            else: vals.append(spearman([f(r,k) for r in rr],y))
        rlin_key='linear_to_actual_ratio'
        cancel_key='defect_cancellation_ratio'
        med_r=np.nanmedian([f(r,rlin_key) for r in rr])
        med_c=np.nanmedian([f(r,cancel_key) for r in rr])
        print('| %d | %s | %.3f | %.3f |' % (T,' | '.join(f'{v:.3f}' for v in vals),med_r,med_c))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--l96'); p.add_argument('--ks'); args=p.parse_args()
    if args.l96:
        rows=read_seed_csvs(Path(args.l96),'diagnostic_campaign.csv')
        if rows: summarize('Lorenz--96 + residual MLP',rows)
    if args.ks:
        rows=read_seed_csvs(Path(args.ks),'diagnostic_campaign.csv')
        if rows: summarize('Kuramoto--Sivashinsky + FNO',rows)

if __name__=='__main__': main()
