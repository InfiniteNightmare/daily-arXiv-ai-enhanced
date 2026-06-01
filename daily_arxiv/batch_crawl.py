#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests
from scrapy import Selector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stream_crawl_enhance import (
    ROOT_DIR,
    batched,
    enhance,
    load_history_ids,
    normalize_arxiv_id,
    parse_list_page,
    parse_categories,
    write_jsonl,
)


EXIT_NO_NEW_CONTENT = 1
EXIT_PROCESSING_ERROR = 2
EXIT_DEFERRED = 3
ARXIV_ABS_URL_TEMPLATE = "https://arxiv.org/abs/{paper_id}"
ARXIV_LIST_URL_TEMPLATE = "https://arxiv.org/list/{category}/new"


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
    parser.add_argument("--metadata-delay-seconds", type=float, default=env_float("ARXIV_API_DELAY_SECONDS", 0.5))
    parser.add_argument("--metadata-retries", type=int, default=env_int("ARXIV_API_NUM_RETRIES", 0, min_value=0))
    parser.add_argument("--metadata-timeout", type=float, default=env_float("ARXIV_API_TIMEOUT_SECONDS", 30.0, min_value=0.1))
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


def pending_metadata_path(output_dir: Path, target_date: str) -> Path:
    return output_dir / "_pending" / f"{target_date}.metadata.jsonl"


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


def checkpoint_pending_metadata(path: Path, seeds: List[Dict], metadata_by_id: Dict[str, Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        metadata_by_id[normalize_arxiv_id(seed["id"])]
        for seed in seeds
        if normalize_arxiv_id(seed["id"]) in metadata_by_id
    ]
    write_jsonl(path, rows)
    print(f"Checkpointed pending metadata={len(rows)}/{len(seeds)} at {path}", file=sys.stderr)


def clear_pending_file(path: Path, label: str):
    if path.exists():
        path.unlink()
        print(f"Cleared pending {label} checkpoint: {path}", file=sys.stderr)


def load_pending_metadata(path: Path, seed_ids: set) -> Dict[str, Dict]:
    metadata_by_id = {}
    for row in read_jsonl(path):
        paper_id = normalize_arxiv_id(row.get("id", ""))
        if paper_id in seed_ids and row.get("title") and row.get("summary"):
            row["id"] = paper_id
            metadata_by_id[paper_id] = row
    if metadata_by_id:
        print(f"Loaded pending metadata={len(metadata_by_id)} from {path}", file=sys.stderr)
    return metadata_by_id


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


def timeout_with_budget(timeout: float, started_at: float, max_seconds: float) -> float:
    ensure_metadata_budget(started_at, max_seconds)
    remaining = metadata_budget_remaining(started_at, max_seconds)
    if remaining is None:
        return timeout
    return max(min(timeout, remaining), 0.1)


def sleep_with_budget(delay: float, started_at: float, max_seconds: float):
    remaining = metadata_budget_remaining(started_at, max_seconds)
    if remaining is not None:
        if remaining <= 0:
            raise CrawlDeferred(
                f"arXiv crawl did not complete within {max_seconds:.0f}s; "
                "pending seed checkpoint was preserved for the next run"
            )
        delay = min(delay, remaining)
    if delay > 0:
        time.sleep(delay)
    ensure_metadata_budget(started_at, max_seconds)


class RequestLimiter:
    def __init__(self, delay_seconds: float):
        self.delay_seconds = max(delay_seconds, 0.0)
        self.last_request_at = 0.0

    def wait(self, started_at: float, max_seconds: float):
        if self.delay_seconds <= 0:
            self.last_request_at = time.monotonic()
            return

        now = time.monotonic()
        wait_seconds = self.last_request_at + self.delay_seconds - now
        if wait_seconds > 0:
            sleep_with_budget(wait_seconds, started_at, max_seconds)
            now = time.monotonic()
        self.last_request_at = now


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def clean_descriptor_text(parts: List[str], descriptor: str) -> str:
    text = clean_text(" ".join(parts))
    return clean_text(re.sub(rf"^{re.escape(descriptor)}\s*", "", text, flags=re.IGNORECASE))


def http_get_with_timeout(session: requests.Session, url: str, timeout: float, **kwargs) -> requests.Response:
    response = session.get(url, timeout=timeout, **kwargs)
    if response.status_code == 200:
        return response

    error = requests.HTTPError(f"HTTP {response.status_code} for {response.url}")
    error.response = response
    raise error


def retry_delay_for_exception(attempt: int, base_seconds: float, exc: Exception) -> float:
    retry_after = enhance.parse_retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, 180.0)
    return retry_delay(attempt, base_seconds)


def crawl_seed_papers_with_budget(
    categories: List[str],
    list_timeout: float,
    started_at: float,
    max_seconds: float,
) -> List[Dict]:
    session = requests.Session()
    target_categories = set(categories)
    seeds = []

    for category in categories:
        url = ARXIV_LIST_URL_TEMPLATE.format(category=category)
        print(f"Fetching arXiv list: {url}", file=sys.stderr)
        attempt = 1
        while True:
            ensure_metadata_budget(started_at, max_seconds)
            try:
                response = http_get_with_timeout(
                    session,
                    url,
                    timeout_with_budget(list_timeout, started_at, max_seconds),
                )
                category_seeds = parse_list_page(response.text, target_categories)
                print(f"Found {len(category_seeds)} candidate papers from {category}", file=sys.stderr)
                seeds.extend(category_seeds)
                break
            except Exception as exc:
                if not enhance.is_retryable_exception(exc):
                    raise

                delay = retry_delay_for_exception(attempt, 1.0, exc)
                print(
                    f"Retry list fetch {category} attempt {attempt} "
                    f"after {delay:.1f}s: {enhance.short_error(exc)}",
                    file=sys.stderr,
                )
                sleep_with_budget(delay, started_at, max_seconds)
                attempt += 1

    return seeds


def parse_abs_page_metadata(seed: Dict, html: str) -> Dict:
    paper_id = normalize_arxiv_id(seed["id"])
    selector = Selector(text=html)

    title = clean_descriptor_text(selector.css("h1.title ::text").getall(), "Title:")
    summary = clean_descriptor_text(selector.css("blockquote.abstract ::text").getall(), "Abstract:")
    authors = [
        clean_text(author)
        for author in selector.css("div.authors a::text").getall()
        if clean_text(author)
    ]

    comment = clean_text(
        " ".join(selector.xpath("//td[contains(@class, 'comments')]//text()").getall())
    ) or None
    subjects_text = clean_text(
        " ".join(selector.xpath("//td[contains(@class, 'subjects')]//text()").getall())
    )
    categories = re.findall(r"\(([^)]+)\)", subjects_text)
    if not categories:
        categories = seed.get("categories", [])

    if not title:
        raise RuntimeError(f"arXiv abs page missing title for {paper_id}")
    if not summary:
        raise RuntimeError(f"arXiv abs page missing abstract for {paper_id}")

    return {
        "id": paper_id,
        "categories": categories,
        "pdf": f"https://arxiv.org/pdf/{paper_id}",
        "abs": f"https://arxiv.org/abs/{paper_id}",
        "authors": authors,
        "title": title,
        "comment": comment,
        "summary": summary,
    }


def metadata_from_batch(
    session: requests.Session,
    seeds: List[Dict],
    timeout: float,
    limiter: RequestLimiter,
    started_at: float,
    max_seconds: float,
) -> List[Dict]:
    enriched = []
    for seed in seeds:
        paper_id = normalize_arxiv_id(seed["id"])
        limiter.wait(started_at, max_seconds)
        response = http_get_with_timeout(
            session,
            ARXIV_ABS_URL_TEMPLATE.format(paper_id=paper_id),
            timeout_with_budget(timeout, started_at, max_seconds),
        )
        enriched.append(parse_abs_page_metadata(seed, response.text))

    return enriched


def resilient_metadata_batch(
    session: requests.Session,
    seeds: List[Dict],
    retry_attempts: int,
    retry_base_seconds: float,
    timeout: float,
    limiter: RequestLimiter,
    started_at: float,
    max_seconds: float,
) -> List[Dict]:
    attempt = 1
    while True:
        ensure_metadata_budget(started_at, max_seconds)
        try:
            return metadata_from_batch(session, seeds, timeout, limiter, started_at, max_seconds)
        except Exception as exc:
            if not enhance.is_retryable_exception(exc):
                raise

            if len(seeds) > 1 and attempt >= retry_attempts:
                delay = retry_delay_for_exception(attempt, retry_base_seconds, exc)
                print(
                    f"Splitting metadata batch of {len(seeds)} after {delay:.1f}s delay: "
                    f"{enhance.short_error(exc)}",
                    file=sys.stderr,
                )
                sleep_with_budget(delay, started_at, max_seconds)
                midpoint = len(seeds) // 2
                left_rows = resilient_metadata_batch(
                    session,
                    seeds[:midpoint],
                    retry_attempts,
                    retry_base_seconds,
                    timeout,
                    limiter,
                    started_at,
                    max_seconds,
                )
                right_rows = resilient_metadata_batch(
                    session,
                    seeds[midpoint:],
                    retry_attempts,
                    retry_base_seconds,
                    timeout,
                    limiter,
                    started_at,
                    max_seconds,
                )
                return left_rows + right_rows

            delay = retry_delay_for_exception(attempt, retry_base_seconds, exc)
            target = f"paper {seeds[0]['id']}" if len(seeds) == 1 else f"batch of {len(seeds)} papers"
            print(
                f"Retry metadata {target} attempt {attempt} after {delay:.1f}s: {enhance.short_error(exc)}",
                file=sys.stderr,
            )
            sleep_with_budget(delay, started_at, max_seconds)
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
    metadata_pending_path = pending_metadata_path(output_dir, args.date)
    if raw_tmp.exists():
        raw_tmp.unlink()

    started_at = time.monotonic()
    history_ids = load_history_ids(output_dir, args.date, args.history_days)
    print(f"Loaded {len(history_ids)} history IDs for deduplication", file=sys.stderr)

    if pending_path.exists():
        unique_seeds = read_jsonl(pending_path)
        print(f"Loaded pending seeds={len(unique_seeds)} from {pending_path}", file=sys.stderr)
    else:
        try:
            seeds = crawl_seed_papers_with_budget(
                categories,
                args.list_timeout,
                started_at,
                args.metadata_max_seconds,
            )
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
        clear_pending_file(pending_path, "seed")
        clear_pending_file(metadata_pending_path, "metadata")
        return EXIT_NO_NEW_CONTENT

    seed_ids = {normalize_arxiv_id(seed["id"]) for seed in unique_seeds}
    metadata_by_id = load_pending_metadata(metadata_pending_path, seed_ids)
    missing_seeds = [
        seed
        for seed in unique_seeds
        if normalize_arxiv_id(seed["id"]) not in metadata_by_id
    ]
    if metadata_by_id:
        print(
            f"Resuming metadata crawl: existing={len(metadata_by_id)}, "
            f"missing={len(missing_seeds)}, total={len(unique_seeds)}",
            file=sys.stderr,
        )

    session = requests.Session()
    limiter = RequestLimiter(args.metadata_delay_seconds)
    batches = list(batched(missing_seeds, max(args.metadata_batch_size, 1)))
    for index, batch in enumerate(batches, 1):
        print(f"Fetching metadata batch {index} with {len(batch)} papers", file=sys.stderr)
        metadata_items = resilient_metadata_batch(
            session,
            batch,
            args.batch_retry_attempts,
            args.batch_retry_base_seconds,
            args.metadata_timeout,
            limiter,
            started_at,
            args.metadata_max_seconds,
        )
        for item in metadata_items:
            metadata_by_id[normalize_arxiv_id(item["id"])] = item
        checkpoint_pending_metadata(metadata_pending_path, unique_seeds, metadata_by_id)

    raw_rows = [
        metadata_by_id[normalize_arxiv_id(seed["id"])]
        for seed in unique_seeds
        if normalize_arxiv_id(seed["id"]) in metadata_by_id
    ]

    if len(raw_rows) != len(unique_seeds):
        raise RuntimeError(f"metadata count mismatch: expected={len(unique_seeds)} actual={len(raw_rows)}")

    write_jsonl(raw_tmp, raw_rows)
    raw_tmp.replace(raw_path)
    clear_pending_file(pending_path, "seed")
    clear_pending_file(metadata_pending_path, "metadata")
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
