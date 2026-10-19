import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "ai"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

import enhance  # noqa: E402


ENGLISH_FIELDS = {
    "tldr": "The paper narrows polynomial gaps.",
    "motivation": "Tree edit distance needs faster algorithms.",
    "method": "A new branching algorithm with memoization.",
    "result": "Improved bounds for bounded height trees.",
    "conclusion": "The gaps are significantly reduced.",
}

CHINESE_FIELDS = {
    "tldr": "本文提出更快的树编辑距离算法。",
    "motivation": "树编辑距离的近似算法存在多项式级差距。",
    "method": "基于分支限界与记忆化的新算法。",
    "result": "在高度受限的树上取得改进界。",
    "conclusion": "显著缩小了已知多项式差距。",
}

MIXED_FIELDS = {
    "tldr": "本文提出 Transformer 变体。",
    "motivation": "attention 计算存在效率瓶颈。",
    "method": "linear attention approximation.",
    "result": "推理加速且精度损失小。",
    "conclusion": "适用于长上下文场景。",
}


class FakeStructure:
    def __init__(self, fields):
        self.fields = fields

    def model_dump(self):
        return dict(self.fields)


def make_item():
    return {
        "id": "2610.00001",
        "title": "Paper title",
        "summary": "Paper abstract",
        "categories": ["cs.AI"],
    }


class TestResponseLanguageMatches(unittest.TestCase):
    def test_english_answer_fails_chinese_target(self):
        self.assertFalse(enhance.response_language_matches(ENGLISH_FIELDS, "Chinese"))

    def test_chinese_answer_passes(self):
        self.assertTrue(enhance.response_language_matches(CHINESE_FIELDS, "Chinese"))

    def test_partial_chinese_passes(self):
        # Fields heavy on English terminology still count as matching.
        self.assertTrue(enhance.response_language_matches(MIXED_FIELDS, "Chinese"))

    def test_non_cjk_target_is_not_checked(self):
        self.assertTrue(enhance.response_language_matches(ENGLISH_FIELDS, "English"))

    def test_blank_language_is_not_checked(self):
        self.assertTrue(enhance.response_language_matches(ENGLISH_FIELDS, ""))


class TestLanguageGuardFlow(unittest.TestCase):
    def run_item(self, invoke_results):
        chain = mock.Mock()
        chain.invoke.side_effect = invoke_results
        item = make_item()
        with mock.patch.object(enhance, "AI_RETRY_ATTEMPTS", 2), mock.patch.object(
            enhance, "AI_RETRY_BASE_SECONDS", 0.0
        ), mock.patch.object(enhance, "AI_MIN_INTERVAL_SECONDS", 0.0):
            enhance.process_single_item(chain, item, "Chinese")
        return item, chain

    def test_english_then_chinese_succeeds_on_retry(self):
        item, chain = self.run_item(
            [FakeStructure(ENGLISH_FIELDS), FakeStructure(CHINESE_FIELDS)]
        )
        self.assertEqual(chain.invoke.call_count, 2)
        self.assertEqual(item["AI_status"], "ok")
        self.assertEqual(item["AI"], CHINESE_FIELDS)

    def test_persistently_english_defers_with_reason(self):
        item, chain = self.run_item(
            [FakeStructure(ENGLISH_FIELDS), FakeStructure(ENGLISH_FIELDS)]
        )
        self.assertEqual(chain.invoke.call_count, 2)
        self.assertEqual(item["AI_status"], "deferred")
        self.assertEqual(item["AI_failure_reason"], "wrong_language")
        # Original metadata stays intact for the next run.
        self.assertEqual(item["id"], "2610.00001")
        self.assertEqual(item["summary"], "Paper abstract")


if __name__ == "__main__":
    unittest.main()
