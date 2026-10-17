"""Restore losslessly compressed spectral CSVs for MATLAB and analysis.

    python prepare_results.py

Idempotent; will not overwrite a differently sized existing CSV.
"""
from pathlib import Path
import gzip, shutil, argparse

p=argparse.ArgumentParser()
p.add_argument('--force',action='store_true',help='Overwrite existing restored CSVs')
a=p.parse_args()
root=Path(__file__).resolve().parent
files=list(root.glob('results/seed_*/spectral_modes_campaign.csv.gz'))
files+=list(root.glob('results_l96_jmlr/seed_*/spectral_modes_campaign.csv.gz'))
files+=list(root.glob('results_ks/seed_*/spectral_modes_campaign.csv.gz'))
if not files: raise SystemExit('No compressed spectral tables found')
for src in files:
    dst=src.with_suffix('')
    if dst.exists() and not a.force:
        print(f'SKIP {dst.relative_to(root)} (exists)')
        continue
    print(f'Restoring {dst.relative_to(root)}')
    with gzip.open(src,'rb') as inp, dst.open('wb') as out:
        shutil.copyfileobj(inp,out,1024*1024)
print('Restored spectral tables.')
