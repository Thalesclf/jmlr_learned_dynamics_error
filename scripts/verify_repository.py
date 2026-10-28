"""Check the saved results in results/current/ and the generated tables and figures.

python scripts/verify_repository.py              structure and consistency of the saved results
python scripts/verify_repository.py --checksums  also compare every file with SHA256SUMS.txt
python scripts/verify_repository.py --legacy     also require the restored historical data
                                                 (docs/DATA_RELEASE.md)

Exits with a non-zero status if a file is missing or inconsistent. This checks saved
outputs and does not repeat any training.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ('l96_e20', 'l96_e150', 'l96_e2000', 'ks_pre')
AUXILIARY = ('l96_e20_ft_one', 'l96_e20_ft_roll', 'l96_e2000_ct_one',
             'l96_e2000_ct_roll', 'ks_one_step', 'ks_rollout')
INTERVENTIONS = ('l96_e20_retry', 'l96_e2000_ct', 'ks_retry')
FIGURES = ('growth', 'scalar', 'budget', 'efficiency', 'coherence', 'spectrum',
           'validity', 'ensemble', 'intervention')


def need(path: Path, problems: list[str]) -> bool:
    if path.is_file():
        return True
    problems.append(f'MISSING: {path.relative_to(ROOT)}')
    return False


def check_current(problems: list[str]) -> None:
    results = ROOT / 'results/current'
    for label in PRIMARY + AUXILIARY:
        for seed in range(5):
            base = results / 'campaign' / label / f'seed_{seed}'
            mp, wp = base / 'meta.json', base / 'windows.csv'
            if need(mp, problems):
                try:
                    meta = json.loads(mp.read_text())
                    if int(meta['seed']) != seed:
                        problems.append(f'Wrong seed in {mp.relative_to(ROOT)}')
                    if label in PRIMARY and int(meta['n_states']) != 100:
                        problems.append(f'Unexpected window count in {mp.relative_to(ROOT)}')
                except (ValueError, KeyError, TypeError) as ex:
                    problems.append(f'Invalid metadata {mp.relative_to(ROOT)}: {ex}')
            if need(wp, problems):
                try:
                    with wp.open(newline='') as f:
                        reader = csv.DictReader(f)
                        if not {'seed', 'start', 'T', 'E', 'Lin', 'Se', 'B', 'Sq', 'gbar', 'eta', 'C', 'N'}.issubset(reader.fieldnames or []):
                            problems.append(f'Missing columns: {wp.relative_to(ROOT)}')
                        else:
                            count = 0
                            horizons = set()
                            max_log_error = 0.0
                            for row in reader:
                                if int(row['seed']) != seed: problems.append(f'Seed mismatch: {wp.relative_to(ROOT)}'); break
                                vals = [float(row[c]) for c in ('E', 'Se', 'gbar', 'eta', 'C', 'N')]
                                if not np.all(np.isfinite(vals)) or min(vals) <= 0:
                                    problems.append(f'Nonpositive/nonfinite budget: {wp.relative_to(ROOT)}'); break
                                product = float(np.prod(vals[1:]))
                                max_log_error = max(max_log_error, abs(np.log(product / vals[0])))
                                count += 1
                                horizons.add(int(row['T']))
                            if count != 900 or len(horizons) != 9:
                                problems.append(f'Unexpected rows/horizons in {wp.relative_to(ROOT)}: {count}/{len(horizons)}')
                            if max_log_error > 1e-7:
                                problems.append(f'Budget factorisation inconsistent: {wp.relative_to(ROOT)}, log error={max_log_error:.2g}')
                except (OSError, ValueError, KeyError, OverflowError) as ex:
                    problems.append(f'Cannot parse {wp.relative_to(ROOT)}: {ex}')
            if label in PRIMARY:
                gp = base / 'terms.csv.gz'
                if need(gp, problems):
                    try:
                        with gzip.open(gp, 'rt', newline='') as f:
                            header = f.readline()
                            if not all(c in header.split(',') for c in ('seed', 'start', 'T', 'k')):
                                problems.append(f'Invalid terms header: {gp.relative_to(ROOT)}')
                            # Ensure gzip CRC/footer is valid (full stream, not only header).
                            for _ in f: pass
                    except (OSError, EOFError, UnicodeError) as ex:
                        problems.append(f'Damaged gzip: {gp.relative_to(ROOT)}: {ex}')
    for label in ('e150', 'e2000'):
        for seed in range(5):
            mp = results / 'l96_models' / label / f'seed_{seed}' / 'model.npz'
            if need(mp, problems):
                try:
                    with np.load(mp, allow_pickle=False) as weights:
                        if not {'net.0.weight', 'net.2.weight', 'net.4.weight', 'mean', 'std'}.issubset(weights.files):
                            problems.append(f'Missing arrays in {mp.relative_to(ROOT)}')
                        for key in weights.files:
                            if not np.all(np.isfinite(weights[key])):
                                problems.append(f'Invalid weights in {mp.relative_to(ROOT)}: {key}')
                except Exception as ex:
                    problems.append(f'Invalid NPZ: {mp.relative_to(ROOT)}: {ex}')
    for label in INTERVENTIONS:
        bp = results / 'intervention_eval' / label / 'paired_summary.csv'
        if need(bp, problems):
            try:
                with bp.open(newline='') as f:
                    rows = list(csv.DictReader(f))
                if len(rows) != 35 or {int(x['seed']) for x in rows} != set(range(5)) or {int(x['h']) for x in rows} != {1, 4, 8, 16, 32, 64, 128} or {int(x['n_windows']) for x in rows} != {312}:
                    problems.append(f'Unexpected intervention layout: {bp.relative_to(ROOT)}')
            except (OSError, ValueError, KeyError) as ex:
                problems.append(f'Invalid intervention CSV: {bp.relative_to(ROOT)}: {ex}')
        need(results / 'intervention_eval' / label / 'window_errors.csv.gz', problems)
    for label in PRIMARY:
        for suffix in (f'../tables/horizon_summary_{label}.csv', f'ensemble/{label}_summary.csv'):
            need(results / suffix if not suffix.startswith('../') else ROOT / 'tables' / suffix.split('/')[-1], problems)
    for key in ('paper_numbers.json', 'tab_intervention.tex', 'tab_representative.tex', 'tab_ensemble.tex'):
        need(ROOT / 'tables' / key, problems)
    for key in FIGURES:
        need(ROOT / 'figures' / f'fig_{key}.pdf', problems)


def check_legacy(problems: list[str]) -> None:
    for dirname in ('results', 'results_l96_jmlr', 'results_ks'):
        for seed in range(5):
            d = ROOT / dirname / f'seed_{seed}'
            for file in ('checkpoint.pt', 'diagnostic_campaign.csv', 'defect_terms_campaign.csv', 'spectral_modes_campaign.csv.gz'):
                need(d / file, problems)
            if dirname == 'results':
                for file in ('perturbation_campaign.csv', 'epsilon_campaign.csv'):
                    need(d / file, problems)
    for dname in ('results_rollout_retry_final', 'results_intervention_final_v3'):
        for syst in ('l96', 'ks'):
            for seed in range(5):
                d = ROOT / dname / syst / f'seed_{seed}'
                for fn in ('config.json', 'results.json'):
                    need(d / fn, problems)
                if dname == 'results_rollout_retry_final':
                    for fn in ('best_one_step.pt', 'best_rollout.pt', 'training_history.csv'):
                        need(d / fn, problems)


def check_manifest(problems: list[str]) -> None:
    manifest = ROOT / 'SHA256SUMS.txt'
    if not manifest.is_file():
        print('No SHA256SUMS.txt found; checksum validation skipped.')
        return
    for ln in manifest.read_text().splitlines():
        if not ln or ln.startswith('#'): continue
        try:
            digest, rel = ln.split('  ', 1)
            target = ROOT / rel
            if not need(target, problems): continue
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                problems.append(f'CHECKSUM MISMATCH: {rel}')
        except ValueError:
            problems.append(f'Invalid manifest entry: {ln!r}')


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--legacy', action='store_true', help='Also require the historical data restored from the release archives')
    pa.add_argument('--checksums', action='store_true', help='Also compare every file with SHA256SUMS.txt')
    args = pa.parse_args()
    errors: list[str] = []
    check_current(errors)
    if args.legacy:
        check_legacy(errors)
    if args.checksums:
        check_manifest(errors)
    if errors:
        print(f'VALIDATION FAILED ({len(errors)} issues):', file=sys.stderr)
        for e in errors:
            print(' - ' + e, file=sys.stderr)
        raise SystemExit(1)
    print('Validation passed: ten five-seed campaign groups, four primary term archives,')
    print('ten L96 model weights, three paired intervention studies, figures and tables'
          + (', historical data' if args.legacy else '') + (', checksums' if args.checksums else '') + '.')
