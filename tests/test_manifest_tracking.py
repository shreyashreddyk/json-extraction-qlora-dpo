import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.manifests import LatestModelManifest, load_latest_model_manifest, save_latest_model_manifest


class ManifestTrackingTest(unittest.TestCase):
    def test_save_and_load_latest_model_manifest(self) -> None:
        with TemporaryDirectory() as repo_dir:
            manifest = LatestModelManifest(
                stage="sft",
                status="success",
                base_model="base-model",
                adapter_path="/drive/adapter",
                schema_version="1.0.0",
                config_paths=["configs/sft.yaml"],
                metrics_paths=["artifacts/metrics/sft_metrics.json"],
                report_paths=["artifacts/reports/sft_report.md"],
            )

            path = save_latest_model_manifest(repo_dir, manifest)
            loaded = load_latest_model_manifest(repo_dir)

            self.assertTrue(path.exists())
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.stage, "sft")
            self.assertEqual(loaded.adapter_path, "/drive/adapter")
