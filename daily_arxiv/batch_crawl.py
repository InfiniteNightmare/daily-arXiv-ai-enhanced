#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import arxiv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stream_crawl_enhance import (
    ROOT_DIR,
    batched,
    crawl_seed_papers,
    enhance,
    load_history_ids,
    normalize_arxiv_id,
    paper_id_from_result,
    parse_categories,
    write_jsonl,
)


EXIT_NO_NEW_CONTENT = 1
EXIT_PROCESSING_ERROR = 2
EXIT_DEFERRED = 3


class CrawlDeferred(RuntimeError):
    pass


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
    parser.add_argument("--metadata-retries", type=int, default=env_int("ARXIV_API_NUM_RETRIES", 0, min_value=0))
    parser.add_argument("--batch-retry-attempts", type=int, default=env_int("ARXIV_BATCH_RETRY_ATTEMPTS", 3))
    parser.add_argument("--batch-retry-base-seconds", type=float, default=env_float("ARXIV_BATCH_RETRY_BASE_SECONDS", 10.0))
    parser.add_argument("--list-retries", type=int, default=env_int("ARXIV_LIST_RETRY_ATTEMPTS", 5))
    parser.add_argument("--list-timeout", type=float, default=env_float("ARXIV_LIST_TIMEOUT_SECONDS", 30.0))
    parser.add_argument("--metadata-max-seconds", type=float, default=env_float("ARXIV_METADATA_MAX_SECONDS", 0.0))
    return parser.parse_args()


def retry_delay(attempt: int, base_seconds: float) -> float:
    return min(base_seconds * (2 ** (attempt - 1)), 180.0)


def pending_seed_path(output_dir: Path, target_date: str) -> Path:
    return output_dir / "_pending" / f"{target_date}.jsonl"


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def checkpoint_pending_seeds(path: Path, seeds: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, seeds)
    print(f"Checkpointed pending seeds={len(seeds)} at {path}", file=sys.stderr)


def clear_pending_seeds(path: Path):
    if path.exists():
        path.unlink()
        print(f"Cleared pending seed checkpoint: {path}", file=sys.stderr)


def metadata_budget_remaining(started_at: float, max_seconds: float):
    if max_seconds <= 0:
        return None
    return max_seconds - (time.monotonic() - started_at)


def ensure_metadata_budget(started_at: float, max_seconds: float):
    remaining = metadata_budget_remaining(started_at, max_seconds)
    if remaining is not None and remaining <= 0:
        raise CrawlDeferred(
            f"arXiv metadata did not complete within {max_seconds:.0f}s; "
            "pending seed checkpoint was preserved for the next run"
        )


def metadata_from_batch(client: arxiv.Client, seeds: List[Dict]) -> List[Dict]:
    seed_by_id = {normalize_arxiv_id(seed["id"]): seed for seed in seeds}
    ids = list(seed_by_id.keys())
    results = list(client.results(arxiv.Search(id_list=ids, max_results=len(ids))))
    result_by_id = {paper_id_from_result(result): result for result in results}
    missing_ids = [paper_id for paper_id in ids if paper_id not in result_by_id]
    if missing_ids:
        raise RuntimeError(f"arXiv metadata response missing IDs: {', '.join(missing_ids)}")

    enriched = []
    for seed in seeds:
        paper_id = normalize_arxiv_id(seed["id"])
        paper = result_by_id[paper_id]

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

    return enriched


def resilient_metadata_batch(
    client: arxiv.Client,
    seeds: List[Dict],
    retry_attempts: int,
    retry_base_seconds: float,
    started_at: float,
    max_seconds: float,
) -> List[Dict]:
    attempt = 1
    while True:
        ensure_metadata_budget(started_at, max_seconds)
        try:
            return metadata_from_batch(client, seeds)
        except Exception as exc:
            if len(seeds) > 1 and attempt >= retry_attempts:
                midpoint = len(seeds) // 2
                print(
                    f"Splitting metadata batch of {len(seeds)} after failure: {enhance.short_error(exc)}",
                    file=sys.stderr,
                )
                left_rows = resilient_metadata_batch(
                    client,
                    seeds[:midpoint],
                    retry_attempts,
                    retry_base_seconds,
                    started_at,
                    max_seconds,
                )
                right_rows = resilient_metadata_batch(
                    client,
                    seeds[midpoint:],
                    retry_attempts,
                    retry_base_seconds,
                    started_at,
                    max_seconds,
                )
                return left_rows + right_rows

            delay = retry_delay(attempt, retry_base_seconds)
            remaining = metadata_budget_remaining(started_at, max_seconds)
            if remaining is not None:
                if remaining <= 0:
                    raise CrawlDeferred(
                        f"arXiv metadata did not complete within {max_seconds:.0f}s; "
                        "pending seed checkpoint was preserved for the next run"
                    ) from exc
                delay = min(delay, remaining)
            target = f"paper {seeds[0]['id']}" if len(seeds) == 1 else f"batch of {len(seeds)} papers"
            print(
                f"Retry metadata {target} attempt {attempt} after {delay:.1f}s: {enhance.short_error(exc)}",
                file=sys.stderr,
            )
            time.sleep(delay)
            attempt += 1


def crawl_raw(args) -> int:
    categories = parse_categories(args.categories)
    if not categories:
        raise ValueError("No arXiv categories configured")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{args.date}.jsonl"
    raw_tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
    pending_path = pending_seed_path(output_dir, args.date)
    if raw_tmp.exists():
        raw_tmp.unlink()

    history_ids = load_history_ids(output_dir, args.date, args.history_days)
    print(f"Loaded {len(history_ids)} history IDs for deduplication", file=sys.stderr)

    if pending_path.exists():
        unique_seeds = read_jsonl(pending_path)
        print(f"Loaded pending seeds={len(unique_seeds)} from {pending_path}", file=sys.stderr)
    else:
        try:
            seeds = crawl_seed_papers(categories, args.list_retries, args.list_timeout)
        except Exception as exc:
            raise CrawlDeferred(
                f"arXiv list pages are temporarily unavailable: {enhance.short_error(exc)}"
            ) from exc

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
        if unique_seeds:
            checkpoint_pending_seeds(pending_path, unique_seeds)

    if not unique_seeds:
        clear_pending_seeds(pending_path)
        return EXIT_NO_NEW_CONTENT

    client = arxiv.Client(
        page_size=max(args.metadata_batch_size, 1),
        delay_seconds=args.metadata_delay_seconds,
        num_retries=args.metadata_retries,
    )

    raw_rows = []
    started_at = time.monotonic()
    for index, batch in enumerate(batched(unique_seeds, max(args.metadata_batch_size, 1)), 1):
        print(f"Fetching metadata batch {index} with {len(batch)} papers", file=sys.stderr)
        metadata_items = resilient_metadata_batch(
            client,
            batch,
            args.batch_retry_attempts,
            args.batch_retry_base_seconds,
            started_at,
            args.metadata_max_seconds,
        )
        raw_rows.extend(metadata_items)

    if len(raw_rows) != len(unique_seeds):
        raise RuntimeError(f"metadata count mismatch: expected={len(unique_seeds)} actual={len(raw_rows)}")

    write_jsonl(raw_tmp, raw_rows)
    raw_tmp.replace(raw_path)
    clear_pending_seeds(pending_path)
    print(f"Wrote raw={len(raw_rows)} metadata_complete=true", file=sys.stderr)
    return 0


def main():
    args = parse_args()
    try:
        exit_code = crawl_raw(args)
    except CrawlDeferred as exc:
        print(f"Batch crawl deferred: {exc}", file=sys.stderr)
        sys.exit(EXIT_DEFERRED)
    except Exception as exc:
        print(f"Batch crawl failed: {exc}", file=sys.stderr)
        sys.exit(EXIT_PROCESSING_ERROR)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
