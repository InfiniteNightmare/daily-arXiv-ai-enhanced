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

import checkpoint_status


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
    def check_single_result(self, ai_result, input_source="abstract"):
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw_path = directory / "2026-08-21.jsonl"
            ai_path = directory / "2026-08-21_AI_enhanced_Chinese.jsonl"
            write_jsonl(raw_path, [{"id": "2608.12345", "summary": "abstract"}])
            write_jsonl(ai_path, [ai_result])

            output = io.StringIO()
            with redirect_stdout(output):
                result = checkpoint_status.check_day(raw_path, ai_path, input_source)
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

    def test_check_day_rejects_ordinary_422_fallback(self):
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

        self.assertFalse(result)
        self.assertIn("incomplete:", output)
        self.assertIn("deferred=1", output)

    def test_check_day_accepts_only_complete_ok_or_legacy_success(self):
        complete_ok = {
            "id": "2608.12345",
            "AI": ai_fields("generated"),
            "AI_status": "ok",
        }
        result, output = self.check_single_result(complete_ok)
        self.assertTrue(result)
        self.assertIn("complete:", output)

        legacy_task_success = {
            "id": "2608.12345",
            "AI": {
                "task": "legacy generated summary",
                "motivation": "generated",
                "method": "generated",
                "result": "generated",
                "conclusion": "generated",
            },
        }
        result, output = self.check_single_result(legacy_task_success, input_source="full")
        self.assertTrue(result)
        self.assertIn("complete:", output)

    def test_check_day_matches_full_input_resume_rule(self):
        explicit_ok_without_source = {
            "id": "2608.12345",
            "AI": ai_fields("generated"),
            "AI_status": "ok",
        }

        result, output = self.check_single_result(
            explicit_ok_without_source,
            input_source="full",
        )

        self.assertFalse(result)
        self.assertIn("deferred=1", output)

        legacy_success = {
            "id": "2608.12345",
            "AI": ai_fields("legacy generated result"),
        }
        result, output = self.check_single_result(legacy_success)
        self.assertTrue(result)
        self.assertIn("complete:", output)

    def test_check_day_rejects_partial_unknown_and_incomplete_results(self):
        cases = {
            "partial": {
                "id": "2608.12345",
                "AI": ai_fields("partial result"),
                "AI_status": "partial",
            },
            "unknown": {
                "id": "2608.12345",
                "AI": ai_fields("unknown result"),
                "AI_status": "unknown",
            },
            "incomplete_ok": {
                "id": "2608.12345",
                "AI": {**ai_fields("generated"), "result": ""},
                "AI_status": "ok",
            },
        }

        for name, item in cases.items():
            with self.subTest(name=name):
                result, output = self.check_single_result(item)
                self.assertFalse(result)
                self.assertIn("deferred=1", output)

    def test_check_day_rejects_legacy_placeholder_failure(self):
        result, output = self.check_single_result(
            {
                "id": "2608.12345",
                "AI": {
                    "tldr": "Summary generation failed",
                    "motivation": "Motivation analysis unavailable",
                    "method": "Method extraction failed",
                    "result": "Result analysis unavailable",
                    "conclusion": "Conclusion extraction failed",
                },
            }
        )

        self.assertFalse(result)
        self.assertIn("deferred=1", output)


if __name__ == "__main__":
    unittest.main()
