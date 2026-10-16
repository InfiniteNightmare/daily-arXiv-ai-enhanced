import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
TO_MD_DIR = ROOT_DIR / "to_md"


def make_row(paper_id, title, categories, ai=None):
    row = {
        "id": paper_id,
        "categories": categories,
        "authors": ["Author"],
        "title": title,
        "summary": f"Summary of {title}",
        "abs": f"https://arxiv.org/abs/{paper_id}",
    }
    if ai is not None:
        row["AI"] = ai
    return row


COMPLETE_AI = {
    "tldr": "a",
    "motivation": "b",
    "method": "c",
    "result": "d",
    "conclusion": "e",
}


class TestConvertRobustness(unittest.TestCase):
    def test_empty_categories_blank_lines_and_incomplete_ai(self):
        with TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "sample.jsonl"
            rows = [
                make_row("2610.00001", "Categorized paper", ["cs.AI"], COMPLETE_AI),
                # Empty categories used to crash the whole conversion with IndexError.
                make_row("2610.00002", "Uncategorized paper", [], COMPLETE_AI),
                # Missing AI block is skipped, not fatal.
                make_row("2610.00003", "No AI yet", ["cs.CL"]),
            ]
            data_file.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "convert.py", "--data", str(data_file)],
                cwd=TO_MD_DIR,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            output = Path(tmp) / "sample.md"
            self.assertTrue(output.exists())
            content = output.read_text(encoding="utf-8")
            self.assertIn("Categorized paper", content)
            self.assertIn("uncategorized", content)
            self.assertIn("Uncategorized paper", content)
            self.assertNotIn("No AI yet", content)

    def test_output_path_derived_from_filename_not_directory(self):
        with TemporaryDirectory() as tmp:
            nested = Path(tmp) / "odd_dir" / "deep"
            nested.mkdir(parents=True)
            data_file = nested / "2026-10-15_AI_enhanced_Chinese.jsonl"
            data_file.write_text(
                json.dumps(make_row("2610.00001", "Paper", ["cs.AI"], COMPLETE_AI),
                           ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "convert.py", "--data", str(data_file)],
                cwd=TO_MD_DIR,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((nested / "2026-10-15.md").exists())


if __name__ == "__main__":
    unittest.main()
