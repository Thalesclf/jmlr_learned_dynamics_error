"""Check the tables of plotted values and the columns that the MATLAB script reads.

scripts/export_matlab_plot_data.py writes the tables; scripts/generate_all_figures.m draws
the figures from them. These checks do not run MATLAB.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/figure_data'
SCRIPT = ROOT / 'scripts/generate_all_figures.m'
ROWS = {'fig_growth': 36, 'fig_scalar': 18, 'fig_budget': 27, 'fig_efficiency': 18, 'fig_spectrum': 16,
        'fig_coherence': 18, 'fig_validity': 68, 'fig_ensemble': 27, 'fig_intervention_change': 105,
        'fig_intervention_factors': 21, 'fig_intervention_amplitude': 63}


def test_figure_tables():
    assert sorted(p.stem for p in DATA.glob('*.csv')) == sorted(ROWS)
    for name, rows in ROWS.items():
        t = pd.read_csv(DATA / f'{name}.csv')
        assert len(t) == rows, name
        assert t['family'].notna().all(), name
        for col in t.columns:      # valid MATLAB names, and no column without values
            assert re.fullmatch(r'[A-Za-z]\w*', col) and t[col].notna().any(), (name, col)
        assert not np.isinf(t.select_dtypes('number').to_numpy(dtype=float)).any(), name   # empty cells are NaN


def test_matlab_script_reads_existing_columns():
    src = SCRIPT.read_text()
    blocks = src.split('\n%% Figure')[1:]
    assert len(blocks) == 9
    for block in blocks:
        files = re.findall(r"readtable\(fullfile\(dataDir, '(\w+)\.csv'\)\)", block)
        columns = set().union(*(pd.read_csv(DATA / f'{f}.csv').columns for f in files))
        used = set(re.findall(r"\b[A-Zd]\.(\w+)\b", block))
        cells = dict(re.findall(r"^(v|f) = \{(.*)\};", block, re.M))
        for name, values in cells.items():
            names = re.findall(r"'(\w+)'", values)
            used |= set(names) if f'd.({name}{{j}})' in block else set()
            used |= {'s_' + n for n in names} if f"d.(['s_' {name}{{j}}])" in block else set()
        assert used and used <= columns, (block.splitlines()[0], used - columns)


def test_matlab_figures_do_not_overwrite_paper_figures():
    src = SCRIPT.read_text()
    assert "outDir = fullfile(root, 'figures', 'matlab');" in src
    names = re.findall(r"exportgraphics\(gcf, fullfile\(outDir, '(fig_\w+)\.pdf'\)", src)
    assert names == ['fig_growth', 'fig_scalar', 'fig_budget', 'fig_efficiency', 'fig_spectrum',
                     'fig_coherence', 'fig_validity', 'fig_ensemble', 'fig_intervention']
    assert 'system(' not in src and 'delete(' not in src
