from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]

def test_saved_five_seed_campaigns_present():
    if not (ROOT/'results_l96_jmlr').is_dir(): pytest.skip('Full scientific datasets not included in code-only checkout')
    for root in ['results_l96_jmlr','results_ks']:
        for seed in range(5):
            p = ROOT / root / f'seed_{seed}'
            assert (p / 'diagnostic_campaign.csv').exists()
            assert (p / 'defect_terms_campaign.csv').exists()
            assert (p / 'spectral_modes_campaign.csv.gz').exists()

def test_archived_intervention_has_both_systems():
    if not (ROOT/'results_intervention_final_v3').is_dir(): pytest.skip('Full scientific datasets not included in code-only checkout')
    for system in ('l96', 'ks'):
        for seed in range(5):
            p = ROOT/'results_intervention_final_v3'/system/f'seed_{seed}'/'results.json'
            assert p.exists()
            records = json.loads(p.read_text())
            assert {r['method'] for r in records} == {'one_step','local_jacobian','rollout','defect_energy'}

def test_single_figure_generator_contains_all_figure_calls():
    matlab=(ROOT/'legacy/matlab/generate_all_figures_historical.m').read_text()
    assert 'fig9_cumulative_spectrum.pdf' in matlab
    assert 'fig10_stress_tests.pdf' in matlab
    assert 'fig11_rollout_intervention.pdf' in matlab
    assert not (ROOT/'generate_rollout_intervention_figures.m').exists()
