import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "ai"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

import checkpoint_status  # noqa: E402


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def ai_fields(details="AI enhancement unavailable"):
    return {
        "tldr": details,
        "motivation": details,
        "method": details,
        "result": details,
        "conclusion": details,
    }


class CheckDayTests(unittest.TestCase):
    def check_single_result(self, ai_result):
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw_path = directory / "2026-08-21.jsonl"
            ai_path = directory / "2026-08-21_AI_enhanced_Chinese.jsonl"
            write_jsonl(raw_path, [{"id": "2608.12345", "summary": "abstract"}])
            write_jsonl(ai_path, [ai_result])

            output = io.StringIO()
            with redirect_stdout(output):
                result = checkpoint_status.check_day(raw_path, ai_path)
            return result, output.getvalue()

    def test_check_day_rejects_legacy_402_fallback(self):
        result, output = self.check_single_result(
            {
                "id": "2608.12345",
                "AI": ai_fields(
                    "AI enhancement failed after retries (llm_error). "
                    "Original arXiv metadata is preserved."
                ),
                "AI_status": "fallback",
                "AI_error": (
                    "Error code: 402 - {'type': 'error', 'error': "
                    "{'type': 'insufficient_balance_error', "
                    "'message': 'Token Plan 用量上限', 'http_code': '402'}}"
                ),
            }
        )

        self.assertFalse(result)
        self.assertIn("incomplete:", output)
        self.assertIn("deferred=1", output)

    def test_check_day_accepts_ordinary_422_fallback(self):
        result, output = self.check_single_result(
            {
                "id": "2608.12345",
                "AI": ai_fields(),
                "AI_status": "fallback",
                "AI_error": (
                    "Error code: 422 - {'type': 'error', 'error': "
                    "{'type': 'unprocessable_entity_error', "
                    "'message': 'input new_sensitive (1026)', 'http_code': '422'}}"
                ),
            }
        )

        self.assertTrue(result)
        self.assertIn("complete:", output)


if __name__ == "__main__":
    unittest.main()
