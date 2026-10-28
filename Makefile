PYTHON ?= python
FAMILIES = l96_e20 l96_e150 l96_e2000 ks_pre
.PHONY: all test verify checksums analysis tables figures figure-data paper overleaf

all: checksums analysis tables figures figure-data verify paper

test:
	$(PYTHON) -m pytest -q
verify:
	$(PYTHON) scripts/verify_repository.py
checksums:
	$(PYTHON) scripts/verify_repository.py --checksums
analysis:
	$(PYTHON) scripts/analysis.py $(FAMILIES)
tables:
	$(PYTHON) scripts/build_tables.py
figures:
	$(PYTHON) scripts/plot_figures.py
figure-data:
	$(PYTHON) scripts/export_matlab_plot_data.py
paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main_jmlr.tex
overleaf:
	$(PYTHON) scripts/export_overleaf.py Overleaf_JMLR.zip
