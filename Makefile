PYTHON ?= python3

.PHONY: install-dev test validate-scaffold tree

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

validate-scaffold:
	$(PYTHON) -m compileall src scripts tests
	$(PYTHON) -m pytest -q

tree:
	find . -maxdepth 3 \
		-not -path './.git*' \
		-not -path './.venv*' \
		-not -path './__pycache__*' \
		| sort

