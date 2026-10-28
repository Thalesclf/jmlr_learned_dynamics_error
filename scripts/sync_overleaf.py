"""Copy the nine figures and nine tables of the manuscript into an Overleaf project folder.

Usage: python scripts/sync_overleaf.py /path/to/overleaf/project
"""
from pathlib import Path
import argparse,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1]
FIGS='growth scalar budget efficiency coherence spectrum validity ensemble intervention'.split()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('overleaf',help='Overleaf source directory with main_jmlr.tex')
    a=p.parse_args()
    dest=Path(a.overleaf).resolve()
    if not (dest/'main_jmlr.tex').is_file():p.error('Not an Overleaf project: missing main_jmlr.tex')
    files=([(ROOT/'figures'/f'fig_{f}.pdf',dest/'figures'/f'fig_{f}.pdf') for f in FIGS]
           +[(p,dest/'tables'/p.name) for p in sorted((ROOT/'tables').glob('tab_*.tex'))])
    assert len(files)==18, 'Expected nine figure PDFs and nine table .tex files'
    for source,target in files:
        if not source.is_file():raise FileNotFoundError(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        if hashlib.sha256(source.read_bytes()).digest()!=hashlib.sha256(target.read_bytes()).digest():
            raise IOError(f'Mismatch after copying {source}')
    print('Synchronized nine vector figures and nine LaTeX tables to',dest)
