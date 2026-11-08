import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "ai"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

import enhance  # noqa: E402

TRUNCATION_MARKER = "[Content truncated for model context]"


class TestFullTextDefaultCap(unittest.TestCase):
    def test_default_cap_is_not_disabled(self):
        # 2026-10/11: default 0 sent book-size full texts untruncated and the
        # provider rejected them with "context window exceeds limit" forever
        # (arXiv 2605.16610). The default must truncate.
        self.assertEqual(enhance.AI_FULL_TEXT_MAX_CHARS, 200_000)

    def test_workflow_shell_default_matches_module_default(self):
        # run.yml always exports AI_FULL_TEXT_MAX_CHARS; when the repo
        # variable is unset its shell-side default wins over the module
        # default. The two defaults must agree (2026-11-08: the workflow
        # default was still "0" and silently disabled the module fix).
        workflow = (ROOT_DIR / ".github" / "workflows" / "run.yml").read_text(
            encoding="utf-8"
        )
        match = __import__("re").search(
            r'if \[ -z "\$ai_full_text_max_chars" \]; then '
            r'ai_full_text_max_chars="(\d+)"; fi',
            workflow,
        )
        self.assertIsNotNone(match, "run.yml ai_full_text_max_chars default not found")
        self.assertEqual(int(match.group(1)), enhance.AI_FULL_TEXT_MAX_CHARS)


class TestTruncateForPrompt(unittest.TestCase):
    def test_short_content_passes_through_untruncated(self):
        # Whitespace is still normalized (3+ newlines collapse), but no
        # truncation marker or content loss may occur under the cap.
        content = "abstract and intro\n\n\n\nconclusion"
        self.assertEqual(
            enhance.truncate_for_prompt(content, 200_000),
            "abstract and intro\n\nconclusion",
        )

    def test_zero_cap_disables_truncation(self):
        content = "x" * 500_000
        self.assertEqual(enhance.truncate_for_prompt(content, 0), content)

    def test_oversized_content_keeps_head_and_tail(self):
        content = ("HEAD" * 60_000) + ("MIDDLE" * 100_000) + ("TAIL" * 60_000)
        truncated = enhance.truncate_for_prompt(content, 200_000)
        self.assertIn(TRUNCATION_MARKER, truncated)
        self.assertTrue(truncated.startswith("HEAD"))
        self.assertTrue(truncated.endswith("TAIL"))
        # head 70% + tail 30% + the marker line.
        self.assertLessEqual(len(truncated), 200_000 + len(TRUNCATION_MARKER) + 4)

    def test_whitespace_collapsed_before_length_check(self):
        content = "a\n\n\n\n\n\n" + "b" * 300
        truncated = enhance.truncate_for_prompt(content, 100)
        self.assertNotIn("\n\n\n", truncated)
        self.assertIn(TRUNCATION_MARKER, truncated)


if __name__ == "__main__":
    unittest.main()
