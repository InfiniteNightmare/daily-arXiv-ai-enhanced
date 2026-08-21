import argparse
import json
import os
import sys
from pathlib import Path

REQUIRED_AI_FIELDS = ("tldr", "motivation", "method", "result", "conclusion")
LEGACY_FAILURE_MARKERS = (
    "ai enhancement unavailable",
    "ai enhancement failed after retries",
    "original arxiv metadata is preserved",
)
LEGACY_FAILURE_FIELD_VALUES = {
    "tldr": "summary generation failed",
    "motivation": "motivation analysis unavailable",
    "method": "method extraction failed",
    "result": "result analysis unavailable",
    "conclusion": "conclusion extraction failed",
}


def has_complete_ai_fields(item: dict) -> bool:
    ai_fields = item.get("AI")
    if not isinstance(ai_fields, dict):
        return False
    return all(
        isinstance(ai_fields.get(field), str) and ai_fields[field].strip()
        for field in REQUIRED_AI_FIELDS
    )


def has_complete_legacy_ai_fields(item: dict) -> bool:
    ai_fields = item.get("AI")
    if not isinstance(ai_fields, dict):
        return False
    summary = ai_fields.get("tldr") or ai_fields.get("task")
    return (
        isinstance(summary, str)
        and bool(summary.strip())
        and all(
            isinstance(ai_fields.get(field), str) and ai_fields[field].strip()
            for field in REQUIRED_AI_FIELDS
            if field != "tldr"
        )
    )


def is_legacy_success(item: dict) -> bool:
    if item.get("AI_error") or item.get("AI_failure_reason"):
        return False
    ai_fields = item.get("AI", {})
    normalized_fields = {
        field: str(
            (ai_fields.get("tldr") or ai_fields.get("task"))
            if field == "tldr"
            else (ai_fields.get(field) or "")
        )
        .strip()
        .lower()
        for field in REQUIRED_AI_FIELDS
    }
    if any(
        normalized_fields[field] == failure_value
        for field, failure_value in LEGACY_FAILURE_FIELD_VALUES.items()
    ):
        return False
    if all(value == "processing failed" for value in normalized_fields.values()):
        return False

    ai_details = json.dumps(ai_fields, ensure_ascii=False).lower()
    return not any(marker in ai_details for marker in LEGACY_FAILURE_MARKERS)


def needs_ai_retry(item: dict) -> bool:
    status = item.get("AI_status")
    if status is None:
        return not has_complete_legacy_ai_fields(item) or not is_legacy_success(item)
    return (
        status != "ok"
        or not has_complete_ai_fields(item)
        or bool(item.get("AI_error") or item.get("AI_failure_reason"))
    )


def is_resumable_ai_result(item: dict, input_source: str) -> bool:
    if not item.get("id") or needs_ai_retry(item):
        return False
    if item.get("AI_status") is None:
        return True
    return not (
        input_source == "full"
        and item.get("AI_input_source") not in {"full", "abstract"}
    )


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def check_day(raw_path: Path, ai_path: Path, input_source: str = "abstract") -> bool:
    raw_ids = []
    for item in read_jsonl(raw_path):
        item_id = item.get("id")
        if item_id and item_id not in raw_ids:
            raw_ids.append(item_id)

    completed_ids = set()
    deferred_ids = set()
    for item in read_jsonl(ai_path):
        item_id = item.get("id")
        if not item_id:
            continue
        if not is_resumable_ai_result(item, input_source):
            deferred_ids.add(item_id)
        else:
            completed_ids.add(item_id)

    completed_ids.difference_update(deferred_ids)
    missing = [item_id for item_id in raw_ids if item_id not in completed_ids]
    if raw_ids and not missing:
        print(f"complete: raw={len(raw_ids)} ai={len(completed_ids)}")
        return True

    print(
        f"incomplete: raw={len(raw_ids)} ai={len(completed_ids)} "
        f"deferred={len(deferred_ids)} missing={len(missing)}"
    )
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check-day")
    check_parser.add_argument("raw_path", type=Path)
    check_parser.add_argument("ai_path", type=Path)
    check_parser.add_argument(
        "--input-source",
        choices=("abstract", "full"),
        default=os.environ.get("AI_INPUT_SOURCE", "abstract"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return 0 if check_day(args.raw_path, args.ai_path, args.input_source) else 1


if __name__ == "__main__":
    sys.exit(main())
