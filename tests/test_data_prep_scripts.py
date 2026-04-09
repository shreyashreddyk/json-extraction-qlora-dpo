import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "data" / "fixtures" / "support_tickets.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class DataPrepScriptsTest(unittest.TestCase):
    def test_prepare_sft_data_writes_prompt_and_messages_manifests(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prompt_output = tmp_path / "sft_prompt_completion.jsonl"
            messages_output = tmp_path / "sft_messages.jsonl"
            summary_output = tmp_path / "sft_summary.json"

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "prepare_sft_data.py"),
                    "--input-path",
                    str(FIXTURE_PATH),
                    "--prompt-completion-output",
                    str(prompt_output),
                    "--messages-output",
                    str(messages_output),
                    "--summary-output",
                    str(summary_output),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            prompt_rows = read_jsonl(prompt_output)
            message_rows = read_jsonl(messages_output)
            summary = json.loads(summary_output.read_text(encoding="utf-8"))

            self.assertEqual(len(prompt_rows), 7)
            self.assertEqual(len(message_rows), 7)
            self.assertEqual(summary["split_counts"], {"eval": 3, "train": 7})
            self.assertEqual(prompt_rows[0]["record_id"], "support-train-001")
            self.assertEqual(
                [message["role"] for message in message_rows[0]["messages"]],
                ["system", "user", "assistant"],
            )

    def test_prepare_eval_data_writes_eval_manifest(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            eval_output = tmp_path / "eval_manifest.jsonl"
            summary_output = tmp_path / "eval_summary.json"

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "prepare_eval_data.py"),
                    "--input-path",
                    str(FIXTURE_PATH),
                    "--output-path",
                    str(eval_output),
                    "--summary-output",
                    str(summary_output),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            eval_rows = read_jsonl(eval_output)
            summary = json.loads(summary_output.read_text(encoding="utf-8"))

            self.assertEqual(len(eval_rows), 3)
            self.assertEqual(summary["eval_record_count"], 3)
            self.assertEqual(eval_rows[0]["record_id"], "support-eval-001")
            self.assertIn("reference_payload", eval_rows[0])
            self.assertIn("reference_json", eval_rows[0])
            self.assertIn("prompt", eval_rows[0])
            self.assertIn("messages", eval_rows[0])
