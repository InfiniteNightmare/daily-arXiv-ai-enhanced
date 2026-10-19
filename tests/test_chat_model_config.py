import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "ai"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

import enhance  # noqa: E402


class TestGetEnvBool(unittest.TestCase):
    def test_blank_value_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"AI_TEST_FLAG": ""}):
            self.assertTrue(enhance.get_env_bool("AI_TEST_FLAG", True))
            self.assertFalse(enhance.get_env_bool("AI_TEST_FLAG", False))
        self.assertTrue(enhance.get_env_bool("AI_TEST_FLAG", True))


class TestChatModelKwargs(unittest.TestCase):
    def test_thinking_disabled_by_default(self):
        with mock.patch.object(enhance, "AI_DISABLE_THINKING", True):
            kwargs = enhance.build_chat_openai_kwargs("deepseek-chat")
        self.assertEqual(kwargs["model"], "deepseek-chat")
        self.assertEqual(kwargs["timeout"], enhance.AI_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})

    def test_thinking_can_be_turned_off(self):
        with mock.patch.object(enhance, "AI_DISABLE_THINKING", False):
            kwargs = enhance.build_chat_openai_kwargs("gpt-4o-mini")
        self.assertNotIn("extra_body", kwargs)

    def test_effort_sent_when_thinking_not_disabled(self):
        with mock.patch.object(enhance, "AI_DISABLE_THINKING", False), mock.patch.object(
            enhance, "AI_THINKING_EFFORT", "low"
        ):
            kwargs = enhance.build_chat_openai_kwargs("MiniMax-M3.1-Flash")
        self.assertEqual(kwargs["extra_body"], {"output_config": {"effort": "low"}})

    def test_disabled_thinking_takes_precedence_over_effort(self):
        with mock.patch.object(enhance, "AI_DISABLE_THINKING", True), mock.patch.object(
            enhance, "AI_THINKING_EFFORT", "low"
        ):
            kwargs = enhance.build_chat_openai_kwargs("MiniMax-M3.1-Flash")
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})


class TestCheckpointThrottle(unittest.TestCase):
    def test_zero_interval_flushes_every_call(self):
        saver = enhance.CheckpointThrottle(0)
        self.assertTrue(saver.should_save())
        self.assertTrue(saver.should_save())

    def test_interval_blocks_immediate_second_flush(self):
        saver = enhance.CheckpointThrottle(3600.0)
        self.assertTrue(saver.should_save())
        self.assertFalse(saver.should_save())

    def test_flushes_again_after_interval_elapses(self):
        saver = enhance.CheckpointThrottle(1.0)
        self.assertTrue(saver.should_save())
        with mock.patch.object(
            enhance.time, "monotonic", return_value=saver.last_saved_at + 2
        ):
            self.assertTrue(saver.should_save())


if __name__ == "__main__":
    unittest.main()
