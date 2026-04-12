from pathlib import Path
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.sampling import select_rows


class SamplingSelectionTest(unittest.TestCase):
    def test_same_seed_returns_same_subset(self) -> None:
        rows = [{"record_id": f"row-{index}"} for index in range(10)]
        first = select_rows(rows, sample_percent=0.3, sample_seed=17)
        second = select_rows(rows, sample_percent=0.3, sample_seed=17)

        self.assertEqual(first.rows, second.rows)
        self.assertEqual(first.metadata.selected_row_count, 3)

    def test_different_seed_changes_subset(self) -> None:
        rows = [{"record_id": f"row-{index}"} for index in range(10)]
        first = select_rows(rows, sample_percent=0.4, sample_seed=17)
        second = select_rows(rows, sample_percent=0.4, sample_seed=18)

        self.assertNotEqual(first.rows, second.rows)

    def test_percent_plus_limit_uses_smaller_subset(self) -> None:
        rows = [{"record_id": f"row-{index}"} for index in range(20)]
        selection = select_rows(rows, sample_percent=0.5, sample_limit=3, sample_seed=17)

        self.assertEqual(selection.metadata.percent_row_count, 10)
        self.assertEqual(selection.metadata.selected_row_count, 3)

    def test_full_percent_keeps_all_rows(self) -> None:
        rows = [{"record_id": f"row-{index}"} for index in range(5)]
        selection = select_rows(rows, sample_percent=1.0, sample_seed=17)

        self.assertEqual(selection.rows, rows)
        self.assertEqual(selection.metadata.sample_mode, "full")

    def test_invalid_percent_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_rows([{"record_id": "row-1"}], sample_percent=0.0, sample_seed=17)


if __name__ == "__main__":
    unittest.main()
