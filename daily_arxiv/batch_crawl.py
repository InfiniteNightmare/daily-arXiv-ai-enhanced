#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import arxiv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stream_crawl_enhance import (
    ROOT_DIR,
    batched,
    crawl_seed_papers,
    enhance,
    fallback_metadata_item,
    load_history_ids,
    normalize_arxiv_id,
    paper_id_from_result,
    parse_categories,
    write_jsonl,
)


def env_int(name: str, default: int, min_value: int = 1) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return max(int(value), min_value)
    except ValueError:
        print(f"Invalid integer for {name}; using {default}", file=sys.stderr)
        return default


def env_float(name: str, default: float, min_value: float = 0.0) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return max(float(value), min_value)
    except ValueError:
        print(f"Invalid float for {name}; using {default}", file=sys.stderr)
        return default


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "data"))
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument("--categories", default=os.environ.get("CATEGORIES", "cs.CV"))
    parser.add_argument("--metadata-batch-size", type=int, default=env_int("ARXIV_METADATA_BATCH_SIZE", 20))
    parser.add_argument("--metadata-delay-seconds", type=float, default=env_float("ARXIV_API_DELAY_SECONDS", 6.0))
    parser.add_argument("--metadata-retries", type=int, default=env_int("ARXIV_API_NUM_RETRIES", 6))
    parser.add_argument("--batch-retry-attempts", type=int, default=env_int("ARXIV_BATCH_RETRY_ATTEMPTS", 3))
    parser.add_argument("--batch-retry-base-seconds", type=float, default=env_float("ARXIV_BATCH_RETRY_BASE_SECONDS", 10.0))
    parser.add_argument("--list-retries", type=int, default=env_int("ARXIV_LIST_RETRY_ATTEMPTS", 5))
    parser.add_argument("--list-timeout", type=float, default=env_float("ARXIV_LIST_TIMEOUT_SECONDS", 30.0))
    return parser.parse_args()


def retry_delay(attempt: int, base_seconds: float) -> float:
    return min(base_seconds * (2 ** (attempt - 1)), 180.0)


def metadata_from_batch(client: arxiv.Client, seeds: List[Dict]) -> Tuple[List[Dict], int]:
    seed_by_id = {normalize_arxiv_id(seed["id"]): seed for seed in seeds}
    ids = list(seed_by_id.keys())
    results = list(client.results(arxiv.Search(id_list=ids, max_results=len(ids))))
    result_by_id = {paper_id_from_result(result): result for result in results}

    enriched = []
    fallback_count = 0
    for seed in seeds:
        paper_id = normalize_arxiv_id(seed["id"])
        paper = result_by_id.get(paper_id)
        if paper is None:
            fallback_count += 1
            enriched.append(fallback_metadata_item(seed))
            continue

        enriched.append({
            "id": paper_id,
            "categories": paper.categories,
            "pdf": f"https://arxiv.org/pdf/{paper_id}",
            "abs": f"https://arxiv.org/abs/{paper_id}",
            "authors": [author.name for author in paper.authors],
            "title": paper.title,
            "comment": paper.comment,
            "summary": paper.summary,
        })

    return enriched, fallback_count


def resilient_metadata_batch(
    client: arxiv.Client,
    seeds: List[Dict],
    retry_attempts: int,
    retry_base_seconds: float,
) -> Tuple[List[Dict], int]:
    last_error = None
    for attempt in range(1, retry_attempts + 1):
        try:
            return metadata_from_batch(client, seeds)
        except Exception as exc:
            last_error = exc
            if attempt >= retry_attempts:
                break
            delay = retry_delay(attempt, retry_base_seconds)
            print(
                f"Retry metadata batch {attempt}/{retry_attempts - 1} "
                f"for {len(seeds)} papers after {delay:.1f}s: {enhance.short_error(exc)}",
                file=sys.stderr,
            )
            time.sleep(delay)

    if len(seeds) > 1:
        midpoint = len(seeds) // 2
        print(
            f"Splitting metadata batch of {len(seeds)} after failure: {enhance.short_error(last_error)}",
            file=sys.stderr,
        )
        left_rows, left_fallback = resilient_metadata_batch(
            client, seeds[:midpoint], retry_attempts, retry_base_seconds
        )
        right_rows, right_fallback = resilient_metadata_batch(
            client, seeds[midpoint:], retry_attempts, retry_base_seconds
        )
        return left_rows + right_rows, left_fallback + right_fallback

    print(f"Metadata failed for {seeds[0]['id']}: {enhance.short_error(last_error)}", file=sys.stderr)
    return [fallback_metadata_item(seeds[0], last_error)], 1


def crawl_raw(args) -> int:
    categories = parse_categories(args.categories)
    if not categories:
        raise ValueError("No arXiv categories configured")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{args.date}.jsonl"
    raw_tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
    if raw_tmp.exists():
        raw_tmp.unlink()

    history_ids = load_history_ids(output_dir, args.date, args.history_days)
    print(f"Loaded {len(history_ids)} history IDs for deduplication", file=sys.stderr)

    seeds = crawl_seed_papers(categories, args.list_retries, args.list_timeout)
    seen_ids = set()
    unique_seeds = []
    duplicate_count = 0
    for seed in seeds:
        paper_id = normalize_arxiv_id(seed["id"])
        if paper_id in seen_ids or paper_id in history_ids:
            duplicate_count += 1
            continue
        seen_ids.add(paper_id)
        seed["id"] = paper_id
        unique_seeds.append(seed)

    print(
        f"Candidates={len(seeds)}, new_after_dedup={len(unique_seeds)}, duplicates={duplicate_count}",
        file=sys.stderr,
    )
    if not unique_seeds:
        return 1

    client = arxiv.Client(
        page_size=max(args.metadata_batch_size, 1),
        delay_seconds=args.metadata_delay_seconds,
        num_retries=args.metadata_retries,
    )

    raw_rows = []
    metadata_fallback_count = 0
    for index, batch in enumerate(batched(unique_seeds, max(args.metadata_batch_size, 1)), 1):
        print(f"Fetching metadata batch {index} with {len(batch)} papers", file=sys.stderr)
        metadata_items, fallback_count = resilient_metadata_batch(
            client,
            batch,
            args.batch_retry_attempts,
            args.batch_retry_base_seconds,
        )
        raw_rows.extend(metadata_items)
        metadata_fallback_count += fallback_count

    if not raw_rows:
        return 1

    write_jsonl(raw_tmp, raw_rows)
    raw_tmp.replace(raw_path)
    print(f"Wrote raw={len(raw_rows)} metadata_fallback={metadata_fallback_count}", file=sys.stderr)
    return 0


def main():
    args = parse_args()
    try:
        exit_code = crawl_raw(args)
    except Exception as exc:
        print(f"Batch crawl failed: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
