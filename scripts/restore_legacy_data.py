"""Restore the historical data archives of release v1.0.0 (see docs/DATA_RELEASE.md).

python scripts/restore_legacy_data.py Historical_reproduction_data_part1.zip
python scripts/restore_legacy_data.py Historical_reproduction_data_part2.zip
python scripts/restore_legacy_data.py --materialize-csv

Every file is checked against the SHA-256 manifest of its archive. --materialize-csv
recreates the uncompressed spectral_modes_campaign.csv files read by some legacy scripts.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, shutil, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LEGACY = ('results/', 'results_l96_jmlr/', 'results_ks/',
          'results_intervention_final_v3/', 'results_rollout_retry_final/')

def restore(archive, target):
    target = Path(target).resolve()
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('HISTORICAL_RELEASE_MANIFEST.json'))
        expected = manifest['files']
        names = {i.filename for i in z.infolist() if not i.is_dir()} - {'HISTORICAL_RELEASE_MANIFEST.json'}
        if names != set(expected):
            raise ValueError(f'Archive and manifest differ: {len(names - set(expected))} unlisted, '
                             f'{len(set(expected) - names)} missing entries')
        for item in z.infolist():
            name = item.filename
            if item.is_dir(): continue
            if name == 'HISTORICAL_RELEASE_MANIFEST.json': continue
            if not name.startswith(LEGACY):
                raise ValueError(f'Unexpected archive entry: {name}')
            dst = (target/name).resolve()
            if not dst.is_relative_to(target):
                raise ValueError(f'Unsafe entry: {name}')
            if dst.exists():
                if hashlib.sha256(dst.read_bytes()).hexdigest() != expected[name]:
                    raise ValueError(f'Existing file is different: {dst}')
                continue
            dst.parent.mkdir(parents=True,exist_ok=True)
            h = hashlib.sha256()
            with z.open(item) as src, dst.open('wb') as out:
                while block:=src.read(2**20):
                    h.update(block);out.write(block)
            if h.hexdigest() != expected[name]:
                dst.unlink(missing_ok=True)
                raise ValueError(f'Archive digest mismatch: {name}')
        print(f"{manifest.get('release', archive)}: {len(expected)} files restored and checked.")

def materialize_csv(target):
    count=0
    for folder in ('results','results_l96_jmlr','results_ks'):
        for gz in sorted((target/folder).glob('seed_*/spectral_modes_campaign.csv.gz')):
            csvpath = gz.with_suffix('')
            if csvpath.is_file(): continue
            with gzip.open(gz,'rb') as src, csvpath.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            count+=1
    print(f'Regenerated {count} missing uncompressed historical spectral CSVs.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('archive',nargs='?',help='Path to Historical_reproduction_data_part1.zip or _part2.zip')
    p.add_argument('--target',default=str(ROOT))
    p.add_argument('--materialize-csv',action='store_true')
    a=p.parse_args()
    if a.archive:restore(a.archive,a.target)
    if a.materialize_csv:materialize_csv(Path(a.target).resolve())
    if not a.archive and not a.materialize_csv:p.error('Provide an archive or --materialize-csv')
