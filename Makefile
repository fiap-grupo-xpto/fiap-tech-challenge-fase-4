
PDF_NOTEBOOK ?= train_model/analise_cancer.ipynb
OUT_BASE = $(basename $(notdir $(PDF_NOTEBOOK)))
# full path to the generated .tex (nbconvert writes into this dir)
LATEX_DIR = ./pdf
LATEX_FILE = $(LATEX_DIR)/$(OUT_BASE).tex
PDF_FILE = $(LATEX_DIR)/$(OUT_BASE).pdf
# base name (without dir or extension) used as jobname for xelatex
LATEX_BASENAME = $(OUT_BASE)
JUPYTER = .venv/bin/jupyter
PYTHON = .venv/bin/python

.PHONY: pdf
pdf: $(LATEX_FILE)
	@echo "Sanitizing LaTeX (remove fancyhdr/header lines)"
	$(PYTHON) tools/generate_pdf.py $(LATEX_FILE)
	@echo "Building PDF (xelatex x3)"
	# run xelatex from the PDF directory so relative image paths (e.g. *_files/) resolve
	(cd $(LATEX_DIR) && xelatex -interaction=nonstopmode -jobname="$(LATEX_BASENAME)" "$(LATEX_BASENAME).tex" >/dev/null || true)
	(cd $(LATEX_DIR) && xelatex -interaction=nonstopmode -jobname="$(LATEX_BASENAME)" "$(LATEX_BASENAME).tex" >/dev/null || true)
	(cd $(LATEX_DIR) && xelatex -interaction=nonstopmode -jobname="$(LATEX_BASENAME)" "$(LATEX_BASENAME).tex" >/dev/null || true)
	@echo "PDF build finished. Check $(PDF_FILE)"

$(LATEX_FILE): $(PDF_NOTEBOOK)
	@echo "Converting notebook to LaTeX"
	# ensure output filename matches OUT_BASE
	$(JUPYTER) nbconvert --to latex --output "$(OUT_BASE)" --output-dir=./pdf $(PDF_NOTEBOOK)
