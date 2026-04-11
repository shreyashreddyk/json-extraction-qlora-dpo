import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.data_registry import SourceGroup, SourceType, load_dataset_registry, registry_by_name


REPO_ROOT = Path(__file__).resolve().parents[1]


class DataRegistryTest(unittest.TestCase):
    def test_load_registry_parses_expected_sources(self) -> None:
        registry = load_dataset_registry(REPO_ROOT / "configs" / "data_sources.yaml")
        indexed = registry_by_name(registry)

        self.assertIn("synthetic_support_tickets_v1", indexed)
        self.assertIn("cfpb_consumer_complaints", indexed)
        self.assertIn("synthetic_hardening_v1", indexed)
        self.assertEqual(indexed["synthetic_support_tickets_v1"].source_type, SourceType.LOCAL_JSONL)
        self.assertEqual(indexed["cfpb_consumer_complaints"].source_group, SourceGroup.DOMAIN_TASK_DATA)
        self.assertFalse(indexed["suneeldk_text_json"].enabled_by_default)

    def test_local_fixture_resolution_uses_repo_relative_path(self) -> None:
        registry = load_dataset_registry(REPO_ROOT / "configs" / "data_sources.yaml")
        indexed = registry_by_name(registry)

        fixture_path = indexed["console_ai_it_helpdesk_synthetic_tickets"].resolve_local_fixture_path(REPO_ROOT)
        self.assertIsNotNone(fixture_path)
        assert fixture_path is not None
        self.assertTrue(fixture_path.exists())


if __name__ == "__main__":
    unittest.main()
