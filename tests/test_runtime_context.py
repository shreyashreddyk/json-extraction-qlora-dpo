import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.runtime import resolve_runtime_context


class RuntimeContextTest(unittest.TestCase):
    def test_resolve_runtime_context_creates_expected_directories(self) -> None:
        with TemporaryDirectory() as repo_dir, TemporaryDirectory() as runtime_dir:
            context = resolve_runtime_context(
                repo_root=repo_dir,
                stage="eval",
                run_name="smoke-test",
                runtime_root=runtime_dir,
            )

            self.assertEqual(context.stage, "eval")
            self.assertTrue(context.metrics_dir.exists())
            self.assertTrue(context.reports_dir.exists())
            self.assertTrue(context.run_dir.exists())

