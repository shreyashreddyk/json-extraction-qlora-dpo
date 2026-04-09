import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.formatting import strip_code_fences
from json_ft.prompts import render_extraction_prompt


class PromptFormattingTest(unittest.TestCase):
    def test_strip_code_fences_handles_json_blocks(self) -> None:
        fenced = "```json\n{\n  \"ok\": true\n}\n```"
        self.assertEqual(strip_code_fences(fenced), '{\n  "ok": true\n}')

    def test_render_extraction_prompt_mentions_schema_fields(self) -> None:
        prompt = render_extraction_prompt("Customer Ava reports a billing issue.")
        self.assertIn("customer_name", prompt)
        self.assertIn("issue_type", prompt)
        self.assertIn("priority", prompt)
