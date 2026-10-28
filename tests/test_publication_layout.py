from pathlib import Path
import hashlib
ROOT = Path(__file__).resolve().parents[1]
FIGS = 'growth scalar budget efficiency coherence spectrum validity ensemble intervention'.split()
def test_paper_assets():
    for x in FIGS:
        assert (ROOT/'figures'/f'fig_{x}.pdf').is_file()
        assert (ROOT/'figures'/f'fig_{x}.pdf').stat().st_size>1000
    assert len(list((ROOT/'tables').glob('tab_*.tex')))==9

def test_original_matlab_source_is_preserved():
    original=ROOT/'legacy/matlab/generate_all_figures_historical.m'
    assert hashlib.sha256(original.read_bytes()).hexdigest()==('376fa4f40444ffe9743e297ebe73adcfa4e1cba907e921f70f964c538fa2f7b0')

def test_figure_scripts():
    assert (ROOT/'scripts/plot_figures.py').is_file()
    assert (ROOT/'scripts/generate_all_figures.m').is_file()   # MATLAB version, writes figures/matlab/
    assert not (ROOT/'legacy/matlab/generate_all_figures.m').exists()
