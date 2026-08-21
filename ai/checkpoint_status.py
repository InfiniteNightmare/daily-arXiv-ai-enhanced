import argparse
import json
import re
import sys
from pathlib import Path

RETRYABLE_HTTP_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 522, 524}
RUN_TERMINAL_HTTP_STATUS_CODES = {401, 402}
LEGACY_TERMINAL_ERROR_MARKERS = (
    "billing_hard_limit_reached",
    "credit_balance_exhausted",
    "insufficient_balance",
    "insufficient_quota",
    "token plan usage limit",
    "token plan 用量上限",
)
LEGACY_DEFERRED_REASONS = (
    "ai_circuit_open",
    "stream_worker_error",
    "time_budget_exceeded",
    "worker_error",
)
STATUS_CODE_PATTERN = re.compile(
    r"(?:error\s+code|status(?:\s+code)?|status_code|http_code)[\"'\s:=]+(\d{3})",
    re.IGNORECASE,
)


def has_metadata_fallback(item: dict) -> bool:
    if item.get("metadata_status") == "fallback":
        return True
    summary = item.get("summary") or ""
    return summary.startswith("arXiv metadata fetch failed after retries.")


def status_code_from_text(error_text: str):
    match = STATUS_CODE_PATTERN.search(error_text)
    return int(match.group(1)) if match else None


def is_legacy_deferred_result(item: dict) -> bool:
    if item.get("AI_status") != "fallback" or has_metadata_fallback(item):
        return False

    error_text = str(item.get("AI_error") or "").lower()
    if any(marker in error_text for marker in LEGACY_TERMINAL_ERROR_MARKERS):
        return True

    status_code = status_code_from_text(error_text)
    if status_code in RUN_TERMINAL_HTTP_STATUS_CODES or status_code in RETRYABLE_HTTP_STATUS_CODES:
        return True
    if status_code is not None and status_code >= 500:
        return True

    ai_details = json.dumps(item.get("AI", {}), ensure_ascii=False).lower()
    return any(f"({reason})" in ai_details for reason in LEGACY_DEFERRED_REASONS)


def needs_ai_retry(item: dict) -> bool:
    return item.get("AI_status") == "deferred" or is_legacy_deferred_result(item)


def is_resumable_ai_result(item: dict, input_source: str) -> bool:
    if not item.get("id") or "AI" not in item or needs_ai_retry(item):
        return False
    if item.get("AI_status") == "fallback":
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


def check_day(raw_path: Path, ai_path: Path) -> bool:
    raw_ids = []
    for item in read_jsonl(raw_path):
        item_id = item.get("id")
        if item_id and item_id not in raw_ids:
            raw_ids.append(item_id)

    completed_ids = set()
    deferred_ids = set()
    for item in read_jsonl(ai_path):
        item_id = item.get("id")
        if not item_id or not item.get("AI"):
            continue
        if needs_ai_retry(item):
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return 0 if check_day(args.raw_path, args.ai_path) else 1


if __name__ == "__main__":
    sys.exit(main())
