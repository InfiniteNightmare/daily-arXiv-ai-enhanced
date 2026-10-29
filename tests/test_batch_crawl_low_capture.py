import argparse
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
DAILY_DIR = ROOT_DIR / "daily_arxiv"
for path in (str(DAILY_DIR),):
    if path not in sys.path:
        sys.path.insert(0, path)

import batch_crawl  # noqa: E402


def make_args(output_dir: Path, date: str = "2026-10-15"):
    return argparse.Namespace(
        date=date,
        output_dir=str(output_dir),
        history_days=7,
        categories="cs.AI,cs.CL",
        metadata_batch_size=50,
        metadata_delay_seconds=0.0,
        metadata_retries=0,
        metadata_timeout=1.0,
        batch_retry_attempts=1,
        batch_retry_base_seconds=0.0,
        list_retries=1,
        list_timeout=1.0,
        metadata_max_seconds=0.0,
        low_capture_zero_fraction=0.5,
        low_capture_fraction=0.4,
        low_capture_max_attempts=3,
    )


def fake_seeds(count: int):
    return [
        {"id": f"2610.1{i:04d}", "categories": ["cs.AI"]} for i in range(1, count + 1)
    ]


def fake_metadata(session, batch, *args, **kwargs):
    return [
        {
            "id": seed["id"],
            "categories": seed.get("categories", []),
            "pdf": f"https://arxiv.org/pdf/{seed['id']}",
            "abs": f"https://arxiv.org/abs/{seed['id']}",
            "authors": ["Author"],
            "title": f"paper {seed['id']}",
            "comment": None,
            "summary": "summary text",
        }
        for seed in batch
    ]


RECENT_HTML = """
<div id='dlpage'>
<ul> <li><a href="/list/cs.AI/recent?skip=0&amp;show=50">Fri, 16 Oct 2026</a></li>
<li><a href="/list/cs.AI/recent?skip=22&amp;show=50">Thu, 15 Oct 2026</a></li></ul>
<dl id='articles'>
<h3>Fri, 16 Oct 2026 (showing 22 of 22 entries )</h3><dt>item 1</dt>
<h3>Thu, 15 Oct 2026 (showing 2 of 2 entries )</h3>
<dt><a href="/abs/2610.33753" title="Abstract">arXiv:2610.33753</a>
  (cross-list from cs.CR)</dt>
<dt><a href="/abs/2610.33380" title="Abstract">arXiv:2610.33380</a>
  (cross-list from cs.AR)</dt>
</dl></div>
"""


class TestRecentSectionParsing(unittest.TestCase):
    def test_section_label_format(self):
        self.assertEqual(batch_crawl.recent_section_label("2026-10-16"), "Fri, 16 Oct 2026")
        # Single-digit days must not be zero-padded (arXiv uses "Mon, 2 Nov").
        self.assertEqual(batch_crawl.recent_section_label("2026-11-02"), "Mon, 2 Nov 2026")

    def test_parse_recent_page_top_and_lower_section(self):
        total, top, excluded = batch_crawl.parse_recent_page(
            RECENT_HTML, "Fri, 16 Oct 2026"
        )
        self.assertEqual(total, 22)
        self.assertEqual(top, "2026-10-16")
        self.assertEqual(excluded, 0)
        total, top, excluded = batch_crawl.parse_recent_page(
            RECENT_HTML, "Thu, 15 Oct 2026"
        )
        self.assertEqual(total, 2)
        self.assertEqual(top, "2026-10-16")
        self.assertEqual(excluded, 0)

    def test_parse_recent_page_excludes_non_target_crosslists(self):
        # 2026-10-27 cs.OS: both section entries were cross-lists from
        # categories outside the target set; /new parsing correctly yields 0
        # candidates for such a section.
        targets = {"cs.OS", "cs.AI"}
        total, top, excluded = batch_crawl.parse_recent_page(
            RECENT_HTML, "Thu, 15 Oct 2026", targets
        )
        self.assertEqual(total, 2)
        self.assertEqual(excluded, 2)

    def test_parse_recent_page_keeps_target_crosslists(self):
        # A cross-list sourced from a target category IS a /new candidate.
        targets = {"cs.OS", "cs.CR"}
        total, top, excluded = batch_crawl.parse_recent_page(
            RECENT_HTML, "Thu, 15 Oct 2026", targets
        )
        self.assertEqual(total, 2)
        self.assertEqual(excluded, 1)

    def test_parse_recent_page_absent_target(self):
        total, top, excluded = batch_crawl.parse_recent_page(
            RECENT_HTML, "Wed, 14 Oct 2026"
        )
        self.assertIsNone(total)
        self.assertEqual(top, "2026-10-16")
        self.assertEqual(excluded, 0)

    def test_parse_recent_section_date(self):
        self.assertEqual(
            batch_crawl.parse_recent_section_date("Fri, 16 Oct 2026"), "2026-10-16"
        )
        self.assertIsNone(batch_crawl.parse_recent_section_date("not a date"))


class TestCaptureInconsistencies(unittest.TestCase):
    TARGET = "2026-10-15"
    TOPS = {"cs.AI": "2026-10-15", "cs.CL": "2026-10-15"}

    def test_rule_a_new_stale_category(self):
        # The 2026-10-15 signature: /recent has the day's section but /new
        # parsed zero candidates for the category.
        issues = batch_crawl.capture_inconsistencies(
            ["cs.AI", "cs.CL"], [0, 17], {"cs.AI": 286, "cs.CL": 177}, self.TOPS, self.TARGET
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("cs.AI", issues[0])
        self.assertIn("286", issues[0])

    def test_rule_b_new_ahead_of_recent(self):
        issues = batch_crawl.capture_inconsistencies(
            ["cs.AI", "cs.CL"], [5, 17], {"cs.AI": None, "cs.CL": 40}, self.TOPS, self.TARGET
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("cs.AI", issues[0])
        self.assertIn("no section", issues[0])

    def test_rule_c_recent_top_is_a_different_day(self):
        issues = batch_crawl.capture_inconsistencies(
            ["cs.AI", "cs.CL"],
            [10, 10],
            {"cs.AI": 100, "cs.CL": 50},
            {"cs.AI": "2026-10-16", "cs.CL": "2026-10-15"},
            self.TARGET,
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("2026-10-16", issues[0])

    def test_steady_state_friday_is_clean(self):
        categories = [
            "cs.DB", "cs.AI", "cs.DC", "cs.SE", "cs.LG", "cs.MA",
            "cs.OS", "cs.CL", "cs.CY", "cs.IR", "cs.DS",
        ]
        counts = [6, 224, 14, 22, 223, 18, 3, 165, 6, 14, 18]
        totals = {
            "cs.DB": 6, "cs.AI": 286, "cs.DC": 16, "cs.SE": 25, "cs.LG": 306,
            "cs.MA": 22, "cs.OS": 3, "cs.CL": 177, "cs.CY": 8, "cs.IR": 14,
            "cs.DS": 21,
        }
        tops = {category: "2026-10-16" for category in categories}
        issues = batch_crawl.capture_inconsistencies(
            categories, counts, totals, tops, "2026-10-16"
        )
        self.assertEqual(issues, [])

    def test_unverifiable_categories_are_skipped(self):
        # cs.AI's /recent page could not be fetched: no evidence, no verdict.
        issues = batch_crawl.capture_inconsistencies(
            ["cs.AI", "cs.CL"], [0, 17], {"cs.CL": 177}, {"cs.CL": self.TARGET}, self.TARGET
        )
        self.assertEqual(issues, [])

    def test_section_of_only_non_target_crosslists_is_not_an_issue(self):
        # 2026-10-27: cs.OS's entire day section was two cross-lists from
        # cs.CR/cs.AR (non-target sources); /new's zero parse was correct.
        issues = batch_crawl.capture_inconsistencies(
            ["cs.OS"],
            [0],
            {"cs.OS": 2},
            {"cs.OS": self.TARGET},
            self.TARGET,
            {"cs.OS": 2},
        )
        self.assertEqual(issues, [])

    def test_partially_excluded_crosslists_still_flag_zero_parse(self):
        # One cross-list from a target source remains as candidate evidence.
        issues = batch_crawl.capture_inconsistencies(
            ["cs.OS"],
            [0],
            {"cs.OS": 2},
            {"cs.OS": self.TARGET},
            self.TARGET,
            {"cs.OS": 1},
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("cs.OS", issues[0])
        self.assertIn("1 non-target cross-lists excluded", issues[0])


class TestIsLowCapture(unittest.TestCase):
    def test_stale_new_page_signature_triggers(self):
        # 2026-10-15: 10 of 11 categories returned zero parseable entries.
        counts = [0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0]
        self.assertTrue(
            batch_crawl.is_low_capture(
                counts, [500], zero_fraction_threshold=0.5, total_fraction_threshold=0.4
            )
        )

    def test_healthy_day_does_not_trigger(self):
        counts = [6, 224, 14, 22, 223, 18, 3, 165, 6, 14, 18]
        self.assertFalse(
            batch_crawl.is_low_capture(
                counts, [478], zero_fraction_threshold=0.5, total_fraction_threshold=0.4
            )
        )

    def test_total_far_below_median_triggers(self):
        self.assertTrue(
            batch_crawl.is_low_capture(
                [10, 12], [500], zero_fraction_threshold=0.5, total_fraction_threshold=0.4
            )
        )

    def test_total_near_median_with_one_zero_category_does_not_trigger(self):
        self.assertFalse(
            batch_crawl.is_low_capture(
                [0, 200, 20], [500], zero_fraction_threshold=0.5, total_fraction_threshold=0.4
            )
        )

    def test_zero_total_is_not_low_capture(self):
        # A day with no announcement exits through the no-new-content path.
        self.assertFalse(
            batch_crawl.is_low_capture(
                [0, 0], [], zero_fraction_threshold=0.5, total_fraction_threshold=0.4
            )
        )

    def test_empty_counts_and_missing_history(self):
        self.assertFalse(
            batch_crawl.is_low_capture(
                [], [500], zero_fraction_threshold=0.5, total_fraction_threshold=0.4
            )
        )
        self.assertFalse(
            batch_crawl.is_low_capture(
                [30], [], zero_fraction_threshold=0.5, total_fraction_threshold=0.4
            )
        )


class TestHistoryRawCounts(unittest.TestCase):
    def test_counts_non_empty_previous_days_only(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "2026-10-14.jsonl").write_text(
                '{"id": "a"}\n{"id": "b"}\n', encoding="utf-8"
            )
            (output_dir / "2026-10-13.jsonl").write_text(
                '{"id": "c"}\n', encoding="utf-8"
            )
            (output_dir / "2026-10-12.jsonl").write_text("", encoding="utf-8")
            # The target date itself must not be counted.
            (output_dir / "2026-10-15.jsonl").write_text(
                '{"id": "d"}\n{"id": "e"}\n{"id": "f"}\n', encoding="utf-8"
            )
            counts = batch_crawl.history_raw_counts(output_dir, "2026-10-15", 7)
            # Most recent day first: 2026-10-14 has 2 rows, 2026-10-13 has 1.
            self.assertEqual(counts, [2, 1])


class TestLowCaptureMarker(unittest.TestCase):
    def test_marker_round_trip(self):
        with TemporaryDirectory() as tmp:
            marker = Path(tmp) / "_pending" / "2026-10-15.low-capture.json"
            self.assertEqual(batch_crawl.read_low_capture_attempts(marker), 0)
            batch_crawl.write_low_capture_marker(marker, 2, 17)
            self.assertEqual(batch_crawl.read_low_capture_attempts(marker), 2)
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8")),
                {"attempts": 2, "last_count": 17},
            )
            batch_crawl.clear_low_capture_marker(marker)
            self.assertEqual(batch_crawl.read_low_capture_attempts(marker), 0)

    def test_corrupt_marker_reads_zero(self):
        with TemporaryDirectory() as tmp:
            marker = Path(tmp) / "marker.json"
            marker.write_text("not json", encoding="utf-8")
            self.assertEqual(batch_crawl.read_low_capture_attempts(marker), 0)


class TestCrawlRawLowCaptureFlow(unittest.TestCase):
    def run_crawl(self, output_dir: Path, seed_counts=None, recent_report=None):
        args = make_args(output_dir)
        if recent_report is None:
            # Steady-state-like report: both categories have today's section.
            recent_report = (
                {"cs.AI": 286, "cs.CL": 177},
                {"cs.AI": "2026-10-15", "cs.CL": "2026-10-15"},
                {"cs.AI": 0, "cs.CL": 0},
            )
        with mock.patch.object(
            batch_crawl,
            "crawl_seed_papers_with_budget",
            return_value=(fake_seeds(17), seed_counts or [0, 17]),
        ) as seed_fetch, mock.patch.object(
            batch_crawl, "load_history_ids", return_value=set()
        ), mock.patch.object(
            batch_crawl, "resilient_metadata_batch", side_effect=fake_metadata
        ), mock.patch.object(
            batch_crawl, "fetch_recent_report", return_value=recent_report
        ) as recent_fetch:
            batch_crawl.crawl_raw(args)
            seed_fetch.assert_called_once()
            recent_fetch.assert_called_once()

    def test_first_low_capture_defers_without_writing_raw(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "2026-10-14.jsonl").write_text(
                '{"id": "x"}\n' * 100, encoding="utf-8"
            )
            with self.assertRaises(batch_crawl.LowCaptureDeferred):
                self.run_crawl(output_dir)  # counts [0, 17]: rule A on cs.AI
            marker = output_dir / "_pending" / "2026-10-15.low-capture.json"
            self.assertEqual(batch_crawl.read_low_capture_attempts(marker), 1)
            # The whole point: a suspicious capture must not be locked in as raw.
            self.assertFalse((output_dir / "2026-10-15.jsonl").exists())
            self.assertFalse(
                (output_dir / "_pending" / "2026-10-15.jsonl").exists()
            )

    def test_second_low_capture_still_defers_and_increments(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "2026-10-14.jsonl").write_text(
                '{"id": "x"}\n' * 100, encoding="utf-8"
            )
            marker = output_dir / "_pending" / "2026-10-15.low-capture.json"
            batch_crawl.write_low_capture_marker(marker, 1, 17)
            with self.assertRaises(batch_crawl.LowCaptureDeferred):
                self.run_crawl(output_dir)
            self.assertEqual(batch_crawl.read_low_capture_attempts(marker), 2)
            self.assertFalse((output_dir / "2026-10-15.jsonl").exists())

    def test_third_attempt_accepts_and_writes_raw(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "2026-10-14.jsonl").write_text(
                '{"id": "x"}\n' * 100, encoding="utf-8"
            )
            marker = output_dir / "_pending" / "2026-10-15.low-capture.json"
            batch_crawl.write_low_capture_marker(marker, 2, 17)
            self.run_crawl(output_dir)
            raw_path = output_dir / "2026-10-15.jsonl"
            self.assertTrue(raw_path.exists())
            rows = [json.loads(line) for line in raw_path.open(encoding="utf-8")]
            self.assertEqual(len(rows), 17)
            self.assertFalse(marker.exists())

    def test_recent_mismatch_alone_defers_even_without_heuristics(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            # Median 30, total 22: no heuristic trigger; rule B on cs.AI only.
            (output_dir / "2026-10-14.jsonl").write_text(
                '{"id": "x"}\n' * 30, encoding="utf-8"
            )
            with self.assertRaises(batch_crawl.LowCaptureDeferred):
                self.run_crawl(
                    output_dir,
                    seed_counts=[5, 17],
                    recent_report=(
                        {"cs.AI": None, "cs.CL": 40},
                        {"cs.AI": "2026-10-15", "cs.CL": "2026-10-15"},
                        {"cs.AI": 0, "cs.CL": 0},
                    ),
                )
            marker = output_dir / "_pending" / "2026-10-15.low-capture.json"
            self.assertEqual(batch_crawl.read_low_capture_attempts(marker), 1)
            self.assertFalse((output_dir / "2026-10-15.jsonl").exists())

    def test_recent_unavailable_heuristic_still_guards(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "2026-10-14.jsonl").write_text(
                '{"id": "x"}\n' * 100, encoding="utf-8"
            )
            # /recent entirely unreachable: counts [0, 17] still defer via the
            # heuristic backstop.
            with self.assertRaises(batch_crawl.LowCaptureDeferred):
                self.run_crawl(output_dir, recent_report=({}, {}, {}))
            self.assertFalse((output_dir / "2026-10-15.jsonl").exists())

    def test_healthy_capture_clears_stale_marker(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "2026-10-14.jsonl").write_text(
                '{"id": "x"}\n' * 100, encoding="utf-8"
            )
            marker = output_dir / "_pending" / "2026-10-15.low-capture.json"
            batch_crawl.write_low_capture_marker(marker, 1, 17)
            # Counts [60, 20]: total 80, median 100, consistent /recent report.
            self.run_crawl(
                output_dir,
                seed_counts=[60, 20],
                recent_report=(
                    {"cs.AI": 100, "cs.CL": 50},
                    {"cs.AI": "2026-10-15", "cs.CL": "2026-10-15"},
                    {"cs.AI": 0, "cs.CL": 0},
                ),
            )
            self.assertFalse(marker.exists())
            self.assertTrue((output_dir / "2026-10-15.jsonl").exists())

    def test_crosslist_only_recent_section_accepts_zero_parse(self):
        # 2026-10-27 end to end: cs.AI's section holds two cross-lists from
        # non-target sources and /new parsed 0 for it, while the other two
        # categories parsed normally and their sections agree. Rule A must not
        # fire for cs.AI, so the crawl is accepted and raw is written on the
        # first attempt. History of 30 keeps both heuristic thresholds quiet
        # (zero fraction 1/3 < 0.5, total 20 >= 0.4*30).
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "2026-10-14.jsonl").write_text(
                '{"id": "x"}\n' * 30, encoding="utf-8"
            )
            args = make_args(output_dir)
            args.categories = "cs.AI,cs.CL,cs.OS"
            recent_report = (
                {"cs.AI": 2, "cs.CL": 40, "cs.OS": 2},
                {
                    "cs.AI": "2026-10-15",
                    "cs.CL": "2026-10-15",
                    "cs.OS": "2026-10-15",
                },
                {"cs.AI": 2, "cs.CL": 0, "cs.OS": 0},
            )
            with mock.patch.object(
                batch_crawl,
                "crawl_seed_papers_with_budget",
                return_value=(fake_seeds(17), [0, 17, 3]),
            ), mock.patch.object(
                batch_crawl, "load_history_ids", return_value=set()
            ), mock.patch.object(
                batch_crawl, "resilient_metadata_batch", side_effect=fake_metadata
            ), mock.patch.object(
                batch_crawl, "fetch_recent_report", return_value=recent_report
            ):
                batch_crawl.crawl_raw(args)
            self.assertTrue((output_dir / "2026-10-15.jsonl").exists())
            self.assertFalse(
                (output_dir / "_pending" / "2026-10-15.low-capture.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
