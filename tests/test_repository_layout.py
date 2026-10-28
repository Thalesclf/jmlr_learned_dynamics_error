from pathlib import Path
from subprocess import run

ROOT = Path(__file__).resolve().parents[1]

def test_current_publication_layout():
    assert (ROOT / "src/realized_error/diagnostics.py").is_file()
    for name in ('growth','scalar','budget','efficiency','spectrum','coherence','validity','ensemble','intervention'):
        assert (ROOT / 'figures' / f'fig_{name}.pdf').is_file()
        assert (ROOT / 'data/processed/figure_data' / f'fig_{name}.csv').is_file() or name == 'intervention'
    assert len(list((ROOT / 'data/processed/figure_data').glob('fig_intervention_*.csv'))) == 3
    assert len(list((ROOT/'tables').glob('tab_*.tex'))) == 9
    assert (ROOT/'paper/main_jmlr.tex').is_file()
    assert (ROOT/'legacy/RENAME_LOG.md').is_file()

def test_no_obsolete_active_data_paths():
    for base in ('scripts','tests','src'):
        for f in (ROOT/base).rglob('*'):
            if f.suffix not in ('.py','.m','.ps1','.sh'): continue
            text = f.read_text(errors='replace')
            # Avoid accidentally embedding an old result root in active code.
            old = 'results' + '_v2/'
            assert old not in text, f
