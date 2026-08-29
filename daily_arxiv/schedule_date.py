import argparse
import re
from datetime import UTC, datetime, time, timedelta

SCHEDULE_PATTERN = re.compile(
    r"^\s*(?P<minute>\d{1,2})\s+(?P<hour>\d{1,2})\s+\*\s+\*\s+1-5\s*$"
)


def parse_schedule(schedule: str) -> tuple[int, int]:
    match = SCHEDULE_PATTERN.fullmatch(schedule)
    if match is None:
        raise ValueError(
            f"Unsupported schedule {schedule!r}; expected '<minute> <hour> * * 1-5'"
        )

    minute = int(match.group("minute"))
    hour = int(match.group("hour"))
    if minute > 59 or hour > 23:
        raise ValueError(f"Invalid schedule time in {schedule!r}")
    return hour, minute


def resolve_scheduled_date(schedule: str, now: datetime | None = None):
    hour, minute = parse_schedule(schedule)
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    else:
        current = current.astimezone(UTC)

    candidate = datetime.combine(
        current.date(),
        time(hour=hour, minute=minute, tzinfo=UTC),
    )
    while candidate > current or candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate.date()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("schedule", help="The exact cron string from github.event.schedule")
    parser.add_argument(
        "--now",
        help="Optional ISO-8601 current time for diagnostics and tests; defaults to now",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.fromisoformat(args.now) if args.now else None
    print(resolve_scheduled_date(args.schedule, now).isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
