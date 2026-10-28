"""Write a standalone Overleaf ZIP of the manuscript, with the figures and tables in local folders.

Example: python scripts/export_overleaf.py Overleaf_JMLR.zip
"""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIGS = ('growth','scalar','budget','efficiency','spectrum','coherence','validity','ensemble','intervention')

def build(target: Path) -> None:
    source = ROOT / 'paper'
    tex = sorted(x for x in source.rglob('*') if x.is_file() and x.suffix in {'.tex', '.bib', '.sty', '.md'})
    figures = [ROOT/'figures'/f'fig_{name}.pdf' for name in FIGS]
    tables = sorted((ROOT/'tables').glob('tab_*.tex'))
    if len(tables) != 9 or any(not p.is_file() for p in figures):
        raise RuntimeError('Missing published figure/table artifacts')
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in tex:
            # The in-repository paper uses sibling output dirs; the standalone Overleaf export uses local output dirs.
            body = p.read_text(encoding='utf-8')
            if p.suffix == '.tex':
                body = body.replace('{../figures/', '{figures/').replace('{../tables/', '{tables/')
            z.writestr(p.relative_to(source).as_posix(), body)
        for p in figures:
            z.write(p, 'figures/'+p.name)
        for p in tables:
            z.write(p, 'tables/'+p.name)
    print(f'Exported {len(tex)} paper sources, {len(figures)} figures, {len(tables)} tables: {target}')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    a=ap.parse_args()
    build(a.output)
