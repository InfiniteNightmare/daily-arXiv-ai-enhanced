import re
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_YML = ROOT_DIR / ".github" / "workflows" / "run.yml"


class TestDedupFreshCrawlGuard(unittest.TestCase):
    """The dedup step must only run check_stats for freshly crawled raw.

    Resumed dates (raw pre-existed or the exit-1 sweep fallback) skip the
    dedup: check_stats deletes a fully-duplicated raw and stops the
    workflow, wedging the backlog sweep on reconstructed weekend dates
    that copy the prior Friday's listing (2026-11-12 x4, 2026-11-15).
    """

    def setUp(self):
        self.text = RUN_YML.read_text(encoding="utf-8")

    def step_script(self, step_name):
        pattern = re.compile(
            r"- name: " + re.escape(step_name) + r"\n(.*?)(?=\n    - name: )",
            re.S,
        )
        match = pattern.search(self.text)
        self.assertIsNotNone(match, f"step {step_name} not found")
        return match.group(1)

    def test_crawl_step_defaults_fresh_crawl_false(self):
        script = self.step_script("Crawl arXiv papers")
        self.assertIn('echo "fresh_crawl=false"', script)
        # Only the batch_crawl exit-0 branch may flip it to true.
        self.assertEqual(script.count('"fresh_crawl=true"'), 1)
        self.assertGreater(script.find("fresh_crawl=false"), -1)
        true_pos = script.find('"fresh_crawl=true"')
        self.assertGreater(
            script.find("case $crawl_exit_code"), -1
        )
        self.assertLess(
            script.find("case $crawl_exit_code"), true_pos,
            "fresh_crawl=true must be inside the exit-code case block",
        )

    def test_raw_exists_branch_does_not_mark_fresh(self):
        script = self.step_script("Crawl arXiv papers")
        skip_branch = script[
            script.find("跳过重新爬取") : script.find("case $crawl_exit_code")
        ]
        self.assertIn("跳过重新爬取", skip_branch)
        self.assertNotIn("fresh_crawl=true", skip_branch)

    def test_sweep_fallback_branch_does_not_mark_fresh(self):
        script = self.step_script("Crawl arXiv papers")
        fallback = script[script.find("no_new_content_swept_backlog") - 800 :][
            :1600
        ]
        self.assertIn("no_new_content_swept_backlog", fallback)
        self.assertNotIn("fresh_crawl=true", fallback)

    def test_dedup_step_skips_resumed_dates(self):
        script = self.step_script("Check for duplicates")
        self.assertIn("fresh_crawl", script)
        bypass_pos = script.find("skipping dedup check")
        check_stats_pos = script.find("check_stats.py")
        self.assertGreater(bypass_pos, -1)
        self.assertLess(bypass_pos, check_stats_pos)
        # The bypass must still propagate has_new_content for the AI step.
        self.assertIn(
            'has_new_content=true', script[:check_stats_pos]
        )


if __name__ == "__main__":
    unittest.main()
