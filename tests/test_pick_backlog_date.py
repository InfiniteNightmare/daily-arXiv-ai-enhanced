import json
import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
DAILY_DIR = ROOT_DIR / "daily_arxiv"
AI_DIR = ROOT_DIR / "ai"
for path in (str(DAILY_DIR), str(AI_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pick_backlog_date  # noqa: E402


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def paper_ids(date_text, count):
    prefix = date_text.replace("-", ".")
    return [f"{prefix}.{str(index).zfill(5)}" for index in range(1, count + 1)]


def ok_row(paper_id, input_source=None):
    row = {
        "id": paper_id,
        "AI_status": "ok",
        "AI": {
            "tldr": "t",
            "motivation": "m",
            "method": "me",
            "result": "r",
            "conclusion": "c",
        },
    }
    if input_source:
        row["AI_input_source"] = input_source
    return row


def deferred_row(paper_id):
    return {
        "id": paper_id,
        "AI_status": "deferred",
        "AI_failure_reason": "ai_circuit_open",
        "AI_error": "Error code: 402",
        "AI": {
            "tldr": "abstract",
            "motivation": "AI enhancement unavailable (ai_circuit_open).",
            "method": "AI enhancement unavailable; see the original abstract.",
            "result": "AI enhancement unavailable; see the original abstract.",
            "conclusion": "AI enhancement unavailable (ai_circuit_open).",
        },
    }


def legacy_ok_row(paper_id):
    return {
        "id": paper_id,
        "AI": {
            "tldr": "legacy tldr",
            "motivation": "m",
            "method": "me",
            "result": "r",
            "conclusion": "c",
        },
    }


def make_raw(data_dir, date_text, count):
    write_jsonl(
        data_dir / f"{date_text}.jsonl",
        [{"id": paper_id} for paper_id in paper_ids(date_text, count)],
    )


class PickBacklogDateTest(unittest.TestCase):
    def pick(self, data_dir, *, lookback_days=7, language="Chinese",
             input_source="full", today=None):
        return pick_backlog_date.pick(
            data_dir,
            lookback_days=lookback_days,
            language=language,
            input_source=input_source,
            today=today,
        )

    def test_picks_date_with_largest_gap(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            for date_text, missing in (("2026-09-28", 3), ("2026-09-29", 10), ("2026-09-30", 5)):
                ids = paper_ids(date_text, 20)
                make_raw(data_dir, date_text, 20)
                rows = [ok_row(paper_id, input_source="full") for paper_id in ids[:20 - missing]]
                rows += [deferred_row(paper_id) for paper_id in ids[20 - missing:]]
                write_jsonl(data_dir / f"{date_text}_AI_enhanced_Chinese.jsonl", rows)
            picked = self.pick(data_dir, today=date(2026, 10, 4))
            self.assertEqual("2026-09-29", picked)

    def test_oldest_date_wins_ties(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            for date_text in ("2026-09-28", "2026-09-30"):
                ids = paper_ids(date_text, 4)
                make_raw(data_dir, date_text, 4)
                write_jsonl(
                    data_dir / f"{date_text}_AI_enhanced_Chinese.jsonl",
                    [ok_row(ids[0], input_source="full"), deferred_row(ids[1]),
                     deferred_row(ids[2]), deferred_row(ids[3])],
                )
            picked = self.pick(data_dir, today=date(2026, 10, 4))
            self.assertEqual("2026-09-28", picked)

    def test_complete_dates_yield_none(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            make_raw(data_dir, "2026-09-28", 2)
            write_jsonl(
                data_dir / "2026-09-28_AI_enhanced_Chinese.jsonl",
                [ok_row(paper_id, input_source="full") for paper_id in paper_ids("2026-09-28", 2)],
            )
            self.assertIsNone(self.pick(data_dir, today=date(2026, 10, 4)))

    def test_missing_enhanced_file_counts_full_gap(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            make_raw(data_dir, "2026-09-29", 6)
            self.assertEqual("2026-09-29", self.pick(data_dir, today=date(2026, 10, 4)))

    def test_dates_without_raw_are_ignored(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            # An incomplete enhanced file without raw data must never be picked.
            write_jsonl(
                data_dir / "2026-09-29_AI_enhanced_Chinese.jsonl",
                [deferred_row("id-1"), deferred_row("id-2")],
            )
            self.assertIsNone(self.pick(data_dir, today=date(2026, 10, 4)))
            # A raw-backed date is picked even when a richer non-raw file exists.
            make_raw(data_dir, "2026-09-30", 1)
            write_jsonl(
                data_dir / "2026-09-30_AI_enhanced_Chinese.jsonl",
                [deferred_row("2026.09.30.00001")],
            )
            self.assertEqual("2026-09-30", self.pick(data_dir, today=date(2026, 10, 4)))

    def test_lookback_excludes_older_dates(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            make_raw(data_dir, "2026-09-20", 4)
            self.assertIsNone(self.pick(data_dir, lookback_days=7, today=date(2026, 10, 4)))
            self.assertEqual(
                "2026-09-20", self.pick(data_dir, lookback_days=30, today=date(2026, 10, 4))
            )

    def test_success_rules_follow_checkpoint_status(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            ids = paper_ids("2026-09-29", 4)
            make_raw(data_dir, "2026-09-29", 4)
            rows = [
                ok_row(ids[0], input_source="full"),
                legacy_ok_row(ids[1]),
                deferred_row(ids[2]),
                {"id": ids[3], "AI_status": "ok",
                 "AI": {"tldr": "", "motivation": "m", "method": "me", "result": "r", "conclusion": "c"}},
            ]
            write_jsonl(data_dir / "2026-09-29_AI_enhanced_Chinese.jsonl", rows)
            missing = pick_backlog_date.count_missing(
                data_dir / "2026-09-29.jsonl",
                data_dir / "2026-09-29_AI_enhanced_Chinese.jsonl",
                "full",
            )
            self.assertEqual(2, missing)

    def test_language_fallback_uses_single_glob_match(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            ids = paper_ids("2026-09-29", 2)
            make_raw(data_dir, "2026-09-29", 2)
            write_jsonl(
                data_dir / "2026-09-29_AI_enhanced_English.jsonl",
                [ok_row(ids[0], input_source="full"), deferred_row(ids[1])],
            )
            picked = self.pick(data_dir, language="Chinese", today=date(2026, 10, 4))
            self.assertEqual("2026-09-29", picked)


if __name__ == "__main__":
    unittest.main()
