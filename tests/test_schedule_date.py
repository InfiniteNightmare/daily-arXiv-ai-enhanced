import sys
import unittest
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DAILY_ARXIV_DIR = ROOT_DIR / "daily_arxiv"
if str(DAILY_ARXIV_DIR) not in sys.path:
    sys.path.insert(0, str(DAILY_ARXIV_DIR))

import schedule_date


class ResolveScheduledDateTests(unittest.TestCase):
    def resolve(self, schedule, timestamp):
        now = datetime.fromisoformat(timestamp)
        return schedule_date.resolve_scheduled_date(schedule, now).isoformat()

    def test_same_day_run_uses_current_utc_date(self):
        self.assertEqual(
            self.resolve("15 14 * * 1-5", "2026-08-28T14:20:56Z"),
            "2026-08-28",
        )

    def test_friday_evening_run_delayed_to_saturday_keeps_friday(self):
        self.assertEqual(
            self.resolve("15 20 * * 1-5", "2026-08-29T02:48:43Z"),
            "2026-08-28",
        )

    def test_friday_morning_run_delayed_to_saturday_keeps_friday(self):
        self.assertEqual(
            self.resolve("15 2 * * 1-5", "2026-08-29T10:00:00Z"),
            "2026-08-28",
        )

    def test_monday_run_does_not_resume_friday(self):
        self.assertEqual(
            self.resolve("15 2 * * 1-5", "2026-08-31T02:20:00Z"),
            "2026-08-31",
        )

    def test_timezone_aware_input_is_normalized_to_utc(self):
        now = datetime(2026, 8, 29, 10, 48, tzinfo=timezone(timedelta(hours=8)))
        self.assertEqual(
            schedule_date.resolve_scheduled_date("15 20 * * 1-5", now).isoformat(),
            "2026-08-28",
        )

    def test_combined_hour_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            schedule_date.resolve_scheduled_date(
                "15 2,8,14,20 * * 1-5",
                datetime(2026, 8, 29, tzinfo=UTC),
            )


if __name__ == "__main__":
    unittest.main()
