PYTHON ?= python3
RSYNC ?= rsync
LOCAL_GOOGLE_DRIVE_ROOT ?= /Users/shreyashreddy/Library/CloudStorage/GoogleDrive-kshreyashreddy@gmail.com/My Drive
DRIVE_SOURCE_DIR ?= $(LOCAL_GOOGLE_DRIVE_ROOT)/json-ft-source
DRIVE_RUNS_DIR ?= $(LOCAL_GOOGLE_DRIVE_ROOT)/json-ft-runs
RSYNC_EXCLUDES := --exclude '__pycache__/' --exclude '*.pyc' --exclude '.ipynb_checkpoints/' --exclude '.DS_Store'

.PHONY: install-dev test validate-scaffold tree eval-baseline drive-init drive-push-source-dry-run drive-push-source drive-pull-artifacts

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

validate-scaffold:
	$(PYTHON) -m compileall src scripts tests
	$(PYTHON) -m pytest -q

eval-baseline:
	$(PYTHON) scripts/eval_model.py \
		--config configs/eval.yaml \
		--run-name baseline-qwen2.5-1.5b \
		--mirror-metrics-to-repo \
		--mirror-report-to-repo \
		--mirror-predictions-to-repo

drive-init:
	mkdir -p "$(DRIVE_SOURCE_DIR)"
	mkdir -p "$(DRIVE_RUNS_DIR)"
	mkdir -p "$(DRIVE_RUNS_DIR)/persistent/metrics"
	mkdir -p "$(DRIVE_RUNS_DIR)/persistent/plots"
	mkdir -p "$(DRIVE_RUNS_DIR)/persistent/reports"
	mkdir -p "$(DRIVE_RUNS_DIR)/persistent/logs"
	mkdir -p "$(DRIVE_RUNS_DIR)/persistent/checkpoints"

drive-push-source-dry-run:
	test -d "$(DRIVE_SOURCE_DIR)" || (echo "Drive source directory does not exist. Run 'make drive-init' first." && exit 1)
	$(RSYNC) -av --delete --dry-run $(RSYNC_EXCLUDES) "./src/" "$(DRIVE_SOURCE_DIR)/src/"
	$(RSYNC) -av --delete --dry-run $(RSYNC_EXCLUDES) "./scripts/" "$(DRIVE_SOURCE_DIR)/scripts/"
	$(RSYNC) -av --delete --dry-run $(RSYNC_EXCLUDES) "./configs/" "$(DRIVE_SOURCE_DIR)/configs/"
	$(RSYNC) -av --delete --dry-run $(RSYNC_EXCLUDES) "./data/" "$(DRIVE_SOURCE_DIR)/data/"
	$(RSYNC) -av --delete --dry-run $(RSYNC_EXCLUDES) "./notebooks/" "$(DRIVE_SOURCE_DIR)/notebooks/"
	$(RSYNC) -av --dry-run $(RSYNC_EXCLUDES) "./requirements-colab.txt" "$(DRIVE_SOURCE_DIR)/requirements-colab.txt"

drive-push-source:
	test -d "$(DRIVE_SOURCE_DIR)" || (echo "Drive source directory does not exist. Run 'make drive-init' first." && exit 1)
	$(RSYNC) -av --delete $(RSYNC_EXCLUDES) "./src/" "$(DRIVE_SOURCE_DIR)/src/"
	$(RSYNC) -av --delete $(RSYNC_EXCLUDES) "./scripts/" "$(DRIVE_SOURCE_DIR)/scripts/"
	$(RSYNC) -av --delete $(RSYNC_EXCLUDES) "./configs/" "$(DRIVE_SOURCE_DIR)/configs/"
	$(RSYNC) -av --delete $(RSYNC_EXCLUDES) "./data/" "$(DRIVE_SOURCE_DIR)/data/"
	$(RSYNC) -av --delete $(RSYNC_EXCLUDES) "./notebooks/" "$(DRIVE_SOURCE_DIR)/notebooks/"
	$(RSYNC) -av $(RSYNC_EXCLUDES) "./requirements-colab.txt" "$(DRIVE_SOURCE_DIR)/requirements-colab.txt"

drive-pull-artifacts:
	mkdir -p "./artifacts/metrics" "./artifacts/plots" "./artifacts/reports" "./artifacts/checkpoints"
	if [ -d "$(DRIVE_SOURCE_DIR)/artifacts/metrics" ]; then $(RSYNC) -av "$(DRIVE_SOURCE_DIR)/artifacts/metrics/" "./artifacts/metrics/"; else echo "No mirrored metrics found in $(DRIVE_SOURCE_DIR)/artifacts/metrics"; fi
	if [ -d "$(DRIVE_SOURCE_DIR)/artifacts/plots" ]; then $(RSYNC) -av "$(DRIVE_SOURCE_DIR)/artifacts/plots/" "./artifacts/plots/"; else echo "No mirrored plots found in $(DRIVE_SOURCE_DIR)/artifacts/plots"; fi
	if [ -d "$(DRIVE_SOURCE_DIR)/artifacts/reports" ]; then $(RSYNC) -av "$(DRIVE_SOURCE_DIR)/artifacts/reports/" "./artifacts/reports/"; else echo "No mirrored reports found in $(DRIVE_SOURCE_DIR)/artifacts/reports"; fi
	if [ -d "$(DRIVE_SOURCE_DIR)/artifacts/checkpoints" ]; then $(RSYNC) -av --include '*/' --include '*.json' --exclude '*' "$(DRIVE_SOURCE_DIR)/artifacts/checkpoints/" "./artifacts/checkpoints/"; else echo "No mirrored checkpoint metadata found in $(DRIVE_SOURCE_DIR)/artifacts/checkpoints"; fi

tree:
	find . -maxdepth 3 \
		-not -path './.git*' \
		-not -path './.venv*' \
		-not -path './__pycache__*' \
		| sort
