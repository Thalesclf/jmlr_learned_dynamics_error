#!/usr/bin/env bash
# Regenerate the tables, figures and manuscript from the saved results in results/current/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python scripts/verify_repository.py --checksums
python scripts/analysis.py l96_e20 l96_e150 l96_e2000 ks_pre
python scripts/build_tables.py
python scripts/plot_figures.py
python scripts/export_matlab_plot_data.py
python scripts/verify_repository.py
if command -v latexmk >/dev/null 2>&1; then
    (cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main_jmlr.tex)
else
    echo 'latexmk not found: tables and figures were generated, the PDF was not compiled' >&2
fi
