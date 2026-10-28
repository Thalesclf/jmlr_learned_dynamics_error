# Building the manuscript

The main file is `main_jmlr.tex`, compiled with pdfLaTeX and BibTeX (`latexmk -pdf main_jmlr.tex`). `jmlr2e.sty` is the JMLR style file.

In the repository, the sources read the figures and tables from `../figures/` and `../tables/`, which are written by `scripts/plot_figures.py` and `scripts/build_tables.py`. `python scripts/export_overleaf.py Overleaf_JMLR.zip` writes a standalone copy of this folder for Overleaf, with the figures and tables in local `figures/` and `tables/` folders, and `python scripts/sync_overleaf.py /path/to/overleaf/project` copies regenerated figures and tables into an existing copy.

Source code and data: https://github.com/Thalesclf/jmlr_learned_dynamics_error
