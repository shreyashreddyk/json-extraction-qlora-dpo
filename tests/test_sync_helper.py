import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.sync import sync_repo_to_runtime


class SyncHelperTest(unittest.TestCase):
    def test_sync_repo_to_runtime_copies_expected_directories(self) -> None:
        with TemporaryDirectory() as repo_dir, TemporaryDirectory() as runtime_dir:
            repo_root = Path(repo_dir)
            (repo_root / "src" / "pkg").mkdir(parents=True)
            (repo_root / "scripts").mkdir()
            (repo_root / "configs").mkdir()
            (repo_root / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (repo_root / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
            (repo_root / "configs" / "test.yaml").write_text("key: value\n", encoding="utf-8")

            result = sync_repo_to_runtime(repo_root, runtime_dir)

            self.assertTrue((Path(runtime_dir) / "src").exists())
            self.assertTrue((Path(runtime_dir) / "scripts").exists())
            self.assertTrue((Path(runtime_dir) / "configs").exists())
            self.assertEqual(len(result.copied_paths), 3)

