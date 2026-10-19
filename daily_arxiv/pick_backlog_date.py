"""Pick the most recent date with incomplete AI results for backlog sweeps.

Scans the data directory from the newest date backwards inside the lookback
window, applying the same success rules as ai/checkpoint_status.py, and
returns the first date with missing papers. Recent content stays fresh while
older gaps are digested with leftover quota windows. Only dates that already
have raw data are considered, so a sweep can never fabricate historical
dates.
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai"))

from checkpoint_status import is_resumable_ai_result, read_jsonl  # noqa: E402


def find_enhanced_file(data_dir: Path, date_text: str, language: str) -> Path | None:
    exact = data_dir / f"{date_text}_AI_enhanced_{language}.jsonl"
    if exact.is_file():
        return exact
    matches = sorted(data_dir.glob(f"{date_text}_AI_enhanced_*.jsonl"))
    if len(matches) == 1:
        return matches[0]
    return None


def count_missing(raw_path: Path, ai_path: Path | None, input_source: str) -> int:
    raw_ids = []
    for item in read_jsonl(raw_path):
        item_id = item.get("id")
        if item_id and item_id not in raw_ids:
            raw_ids.append(item_id)
    if not raw_ids:
        return 0

    if ai_path is None or not ai_path.is_file() or ai_path.stat().st_size == 0:
        return len(raw_ids)

    completed_ids = set()
    deferred_ids = set()
    for item in read_jsonl(ai_path):
        item_id = item.get("id")
        if not item_id:
            continue
        if is_resumable_ai_result(item, input_source):
            completed_ids.add(item_id)
        else:
            deferred_ids.add(item_id)
    completed_ids.difference_update(deferred_ids)
    return sum(1 for item_id in raw_ids if item_id not in completed_ids)


def pick(
    data_dir: Path,
    *,
    lookback_days: int,
    language: str,
    input_source: str,
    today: date | None = None,
) -> str | None:
    current = today or datetime.now(timezone.utc).date()
    if isinstance(current, str):
        current = date.fromisoformat(current)
    lookback_days = max(lookback_days, 0)

    candidates_scanned = 0
    for offset in range(0, lookback_days + 1):
        date_text = (current - timedelta(days=offset)).isoformat()
        raw_path = data_dir / f"{date_text}.jsonl"
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            continue
        try:
            ai_path = find_enhanced_file(data_dir, date_text, language)
            missing = count_missing(raw_path, ai_path, input_source)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(
                f"warning: skipping {date_text}: unreadable data file ({exc})",
                file=sys.stderr,
            )
            continue
        print(f"sweep candidate {date_text}: missing={missing}", file=sys.stderr)
        candidates_scanned += 1
        if missing > 0:
            # Newest incomplete date first: recent content stays complete while
            # older gaps are digested with leftover quota windows.
            return date_text
    if candidates_scanned:
        print(
            f"sweep scanned {candidates_scanned} dates; all complete",
            file=sys.stderr,
        )
    return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory holding data-branch files, e.g. data/",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="How many recent UTC days to scan, including today (default: 7)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Enhanced-file language suffix (defaults to $LANGUAGE or 'Chinese')",
    )
    parser.add_argument(
        "--input-source",
        choices=("abstract", "full"),
        default=None,
        help="AI input source for the success rules (defaults to $AI_INPUT_SOURCE or 'abstract')",
    )
    parser.add_argument(
        "--today",
        default=None,
        help="Override the current UTC date (YYYY-MM-DD); for tests and diagnostics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    language = args.language or os.environ.get("LANGUAGE") or "Chinese"
    input_source = args.input_source or os.environ.get("AI_INPUT_SOURCE") or "abstract"
    picked = pick(
        args.data_dir,
        lookback_days=args.lookback_days,
        language=language,
        input_source=input_source,
        today=args.today,
    )
    if picked:
        print(picked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
