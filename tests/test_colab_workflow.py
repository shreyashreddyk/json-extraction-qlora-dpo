import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class ColabWorkflowContractTest(unittest.TestCase):
    def test_makefile_exposes_clean_slate_drive_targets(self) -> None:
        makefile_text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("drive-init:", makefile_text)
        self.assertIn("drive-reset-source:", makefile_text)
        self.assertIn("drive-reset-runs:", makefile_text)
        self.assertIn("drive-rewrite-colab:", makefile_text)
        self.assertIn('mkdir -p "$(DRIVE_RUNS_DIR)/raw-data"', makefile_text)
        self.assertIn('./README.md" "$(DRIVE_SOURCE_DIR)/README.md"', makefile_text)

    def test_colab_setup_notebook_bootstraps_clean_slate_runtime(self) -> None:
        notebook = json.loads((REPO_ROOT / "notebooks" / "00_colab_setup.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("build_dataset_manifests.py", source)
        self.assertIn("source_dirs_to_create", source)
        self.assertIn("runtime_dirs_to_create", source)
        self.assertNotIn("support_tickets_canonical.jsonl", source)

    def test_stage_notebooks_use_portable_full_profile(self) -> None:
        sft_notebook = json.loads((REPO_ROOT / "notebooks" / "02_sft_review.ipynb").read_text(encoding="utf-8"))
        preference_notebook = json.loads((REPO_ROOT / "notebooks" / "03_preference_pair_audit.ipynb").read_text(encoding="utf-8"))
        dpo_notebook = json.loads((REPO_ROOT / "notebooks" / "04_dpo_review.ipynb").read_text(encoding="utf-8"))

        sft_source = "\n".join("".join(cell.get("source", [])) for cell in sft_notebook["cells"])
        preference_source = "\n".join("".join(cell.get("source", [])) for cell in preference_notebook["cells"])
        dpo_source = "\n".join("".join(cell.get("source", [])) for cell in dpo_notebook["cells"])

        self.assertIn("PROFILE = 'full'", sft_source)
        self.assertIn("TRAIN_SAMPLE_PERCENT", sft_source)
        self.assertIn("--train-sample-percent", sft_source)
        self.assertIn("'full'", preference_source)
        self.assertIn("PAIR_SOURCE_SAMPLE_PERCENT", preference_source)
        self.assertIn("PREFERENCE_BATCH_SIZE", preference_source)
        self.assertIn("--sample-percent", preference_source)
        self.assertIn("--inference-batch-size", preference_source)
        self.assertIn("'full'", dpo_source)
        self.assertIn("DPO_TRAIN_BATCH_SIZE", dpo_source)
        self.assertIn("DPO_GRADIENT_ACCUMULATION_STEPS", dpo_source)
        self.assertIn("--per-device-train-batch-size", dpo_source)
        self.assertIn("--gradient-accumulation-steps", dpo_source)
        self.assertIn("DPO_TRAIN_SAMPLE_PERCENT", dpo_source)
        self.assertIn("--train-sample-percent", dpo_source)
        self.assertIn("sft-full-colab", sft_source)
        self.assertIn("pref-full-colab", preference_source)
        self.assertIn("dpo-full-colab", dpo_source)


if __name__ == "__main__":
    unittest.main()
