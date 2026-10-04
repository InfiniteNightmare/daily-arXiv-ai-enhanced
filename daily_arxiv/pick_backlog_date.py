"""Pick the recent date with the largest AI-enhancement gap for weekend sweeps.

Scans the data directory for dates inside the lookback window that have raw
arXiv data (data/<date>.jsonl) but incomplete AI results, applying the same
success rules as ai/checkpoint_status.py. Prints the date with the largest
number of missing papers (oldest wins ties); prints nothing when every
scanned date is complete. Only dates that already have raw data are
considered, so a sweep can never fabricate historical dates.
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

    candidates = []
    for offset in range(lookback_days, -1, -1):
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
        if missing > 0:
            candidates.append((date_text, missing))

    if not candidates:
        return None
    # Candidates are ordered oldest -> newest, so max() keeps the oldest date
    # among ties.
    best_date, _ = max(candidates, key=lambda entry: entry[1])
    return best_date


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
