# ===================================================================
# ML & Signal Processing on Neural Signals 101 — Makefile
#
#   make setup        Create a Python 3.11 .venv and install everything
#   make notebooks    Build notebooks/*.ipynb from notebooks/_src/*.py
#   make test         Run pytest (unit tests + fast notebook smoke test)
#   make run-all      Execute every notebook top-to-bottom (full data)
#   make headline     Regenerate docs/headline.png
#   make lint         Run ruff
#   make clean        Remove caches and build artifacts
# ===================================================================
.DEFAULT_GOAL := help
VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PY311   := python3.11

# CI/smoke can export NEURO101_SMOKE=1 to shrink data; locally we run fuller.
export PYTHONWARNINGS ?= ignore

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ---- environment --------------------------------------------------
.PHONY: setup
setup: ## Create a Python 3.11 virtualenv and install pinned deps + the package
	@command -v $(PY311) >/dev/null 2>&1 || { \
	  echo "ERROR: python3.11 not found."; \
	  echo "  macOS:  brew install python@3.11"; \
	  echo "  Linux:  sudo apt-get install python3.11 python3.11-venv"; \
	  echo "  or use pyenv: pyenv install 3.11 && pyenv local 3.11"; \
	  exit 1; }
	$(PY311) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel setuptools
	# CPU-only torch from the dedicated index keeps the install small (no CUDA).
	$(PIP) install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "✅ setup complete. Activate with: source $(VENV)/bin/activate"

# ---- notebooks ----------------------------------------------------
.PHONY: notebooks
notebooks: ## Build .ipynb from the percent-format sources in notebooks/_src
	$(PY) scripts/build_notebooks.py

.PHONY: run-all
run-all: notebooks ## Execute every notebook end-to-end (uses full-size data)
	$(PY) scripts/run_all_notebooks.py

# ---- quality ------------------------------------------------------
.PHONY: test
test: notebooks ## Run unit tests + a fast (smoke-mode) notebook execution test
	NEURO101_SMOKE=1 $(PY) -m pytest -q

.PHONY: test-fast
test-fast: ## Run only unit tests, skipping anything that needs a download
	$(PY) -m pytest -q -m "not network and not slow" tests/test_eval.py tests/test_features.py tests/test_preprocessing.py

.PHONY: lint
lint: ## Run ruff over src/, tests/ and scripts/
	$(PY) -m ruff check src tests scripts

.PHONY: headline
headline: ## Regenerate the README headline figure
	$(PY) scripts/make_headline_figure.py

.PHONY: social
social: ## Regenerate the GitHub social-preview banner (docs/social-preview.png)
	$(PY) scripts/make_social_preview.py

# ---- housekeeping -------------------------------------------------
.PHONY: clean
clean: ## Remove caches and build artifacts (keeps the .venv and downloads)
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info build dist
	find . -name '*.pyc' -delete
	find . -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} +
