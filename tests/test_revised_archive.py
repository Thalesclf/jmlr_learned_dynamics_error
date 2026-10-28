"""Fail CI when the claimed current revised datasets have not been committed."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_revised_snapshot_has_all_primary_seed_outputs():
    for family in ('l96_e20', 'l96_e150', 'l96_e2000', 'ks_pre'):
        for seed in range(5):
            directory = ROOT / 'results/current' / 'campaign' / family / f'seed_{seed}'
            assert (directory / 'meta.json').is_file(), f'missing revised metadata: {directory}'
            assert (directory / 'windows.csv').is_file(), f'missing revised windows: {directory}'
            assert (directory / 'terms.csv.gz').is_file(), f'missing revised propagated terms: {directory}'
            assert json.loads((directory / 'meta.json').read_text())['seed'] == seed


def test_revised_tables_and_figures_and_reference_metadata():
    assert (ROOT / 'tables' / 'paper_numbers.json').is_file()
    for name in ('growth', 'scalar', 'budget', 'efficiency', 'coherence', 'spectrum',
                 'validity', 'ensemble', 'intervention'):
        assert (ROOT / 'figures' / f'fig_{name}.pdf').is_file()
    url = 'https://github.com/Thalesclf/jmlr_learned_dynamics_error'
    assert url in (ROOT / 'README.md').read_text()
    assert url in (ROOT / 'CITATION.cff').read_text()
