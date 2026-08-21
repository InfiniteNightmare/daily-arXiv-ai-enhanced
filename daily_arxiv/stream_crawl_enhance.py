#!/usr/bin/env python3
import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Dict, Iterable, List, Tuple

import arxiv
import requests
from scrapy import Selector


ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "ai"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

import enhance  # noqa: E402


RETRYABLE_HTTP_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 522, 524}


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
    parser.add_argument("--language", default=os.environ.get("LANGUAGE", "Chinese"))
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "deepseek-chat"))
    parser.add_argument("--ai-workers", type=int, default=env_int("AI_MAX_WORKERS", 1))
    parser.add_argument("--metadata-batch-size", type=int, default=env_int("ARXIV_METADATA_BATCH_SIZE", 50))
    parser.add_argument("--metadata-delay-seconds", type=float, default=env_float("ARXIV_API_DELAY_SECONDS", 3.0))
    parser.add_argument("--metadata-retries", type=int, default=env_int("ARXIV_API_NUM_RETRIES", 8))
    parser.add_argument("--list-retries", type=int, default=env_int("ARXIV_LIST_RETRY_ATTEMPTS", 5))
    parser.add_argument("--list-timeout", type=float, default=env_float("ARXIV_LIST_TIMEOUT_SECONDS", 30.0))
    return parser.parse_args()


def normalize_arxiv_id(value: str) -> str:
    paper_id = value.strip().rstrip("/")
    paper_id = paper_id.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", paper_id)


def parse_categories(raw_categories: str) -> List[str]:
    return [category.strip() for category in raw_categories.split(",") if category.strip()]


def ordered_target_categories(categories: List[str], target_categories: List[str]) -> List[str]:
    target_category_set = set(target_categories)
    seen = set()
    ordered = []
    for category in categories or []:
        if category in target_category_set and category not in seen:
            ordered.append(category)
            seen.add(category)
    return ordered


def jsonl_rows_from_git(ref_path: str) -> Iterable[Dict]:
    try:
        proc = subprocess.run(
            ["git", "show", ref_path],
            cwd=ROOT_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []

    rows = []
    for line in proc.stdout.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def jsonl_rows_from_file(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_history_ids(output_dir: Path, target_date: str, history_days: int) -> set:
    history_ids = set()
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")

    for offset in range(1, history_days + 1):
        date_str = (target_dt - timedelta(days=offset)).strftime("%Y-%m-%d")
        rows = list(jsonl_rows_from_file(output_dir / f"{date_str}.jsonl"))
        if not rows:
            rows = list(jsonl_rows_from_git(f"origin/data:data/{date_str}.jsonl"))
        for row in rows:
            paper_id = row.get("id")
            if paper_id:
                history_ids.add(normalize_arxiv_id(paper_id))

    return history_ids


def response_retry_delay(attempt: int) -> float:
    return min(2 ** (attempt - 1), 60.0)


def fetch_text_with_retries(session: requests.Session, url: str, attempts: int, timeout: float) -> str:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.text
            if response.status_code not in RETRYABLE_HTTP_STATUS_CODES:
                response.raise_for_status()

            last_error = requests.HTTPError(f"HTTP {response.status_code} for {url}")
            last_error.response = response
            raise last_error
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            delay = response_retry_delay(attempt)
            print(f"Retry list fetch {attempt}/{attempts - 1} after {delay:.1f}s: {exc}", file=sys.stderr)
            time.sleep(delay)

    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts: {last_error}")


def parse_list_page(html: str, target_categories: List[str]) -> List[Dict]:
    selector = Selector(text=html)
    anchors = []
    for li in selector.css("div[id=dlpage] ul li"):
        href = li.css("a::attr(href)").get()
        if href and "item" in href:
            anchors.append(int(href.split("item")[-1]))

    papers = []
    for paper in selector.css("dl dt"):
        paper_anchor = paper.css("a[name^='item']::attr(name)").get()
        if not paper_anchor:
            continue

        paper_anchor_id = int(paper_anchor.split("item")[-1])
        if anchors and paper_anchor_id >= anchors[-1]:
            continue

        abstract_link = paper.css("a[title='Abstract']::attr(href)").get()
        if not abstract_link:
            continue

        arxiv_id = normalize_arxiv_id(abstract_link)
        paper_dd = paper.xpath("following-sibling::dd[1]")
        primary_subjects_text = " ".join(
            part.strip()
            for part in paper_dd.css(".list-subjects .primary-subject::text").getall()
        )
        primary_categories = re.findall(r"\(([^)]+)\)", primary_subjects_text)
        if primary_categories:
            primary_category = primary_categories[0]
            if primary_category in target_categories:
                papers.append({"id": arxiv_id, "categories": [primary_category]})
            continue

        subjects_text = " ".join(part.strip() for part in paper_dd.css(".list-subjects ::text").getall())
        all_categories = re.findall(r"\(([^)]+)\)", subjects_text)
        matched_categories = ordered_target_categories(all_categories, target_categories)
        if matched_categories:
            papers.append({"id": arxiv_id, "categories": matched_categories})
        elif not all_categories:
            papers.append({"id": arxiv_id, "categories": []})

    return papers


def crawl_seed_papers(categories: List[str], list_retries: int, list_timeout: float) -> List[Dict]:
    session = requests.Session()
    seeds = []

    for category in categories:
        url = f"https://arxiv.org/list/{category}/new"
        print(f"Fetching arXiv list: {url}", file=sys.stderr)
        html = fetch_text_with_retries(session, url, list_retries, list_timeout)
        category_seeds = parse_list_page(html, categories)
        print(f"Found {len(category_seeds)} candidate papers from {category}", file=sys.stderr)
        seeds.extend(category_seeds)

    return seeds


def batched(items: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def paper_id_from_result(result) -> str:
    if hasattr(result, "get_short_id"):
        return normalize_arxiv_id(result.get_short_id())
    return normalize_arxiv_id(getattr(result, "entry_id", ""))


def fallback_metadata_item(seed: Dict, exc: Exception = None) -> Dict:
    item = {
        "id": seed["id"],
        "categories": seed.get("categories", []),
        "pdf": f"https://arxiv.org/pdf/{seed['id']}",
        "abs": f"https://arxiv.org/abs/{seed['id']}",
        "authors": [],
        "title": seed["id"],
        "comment": None,
        "summary": "arXiv metadata fetch failed after retries. Original paper links and ID are preserved.",
        "metadata_status": "fallback",
    }
    if exc is not None:
        item["metadata_error"] = enhance.short_error(exc)
    return item


def enrich_metadata_batch(
    client: arxiv.Client,
    seeds: List[Dict],
    target_categories: List[str],
) -> Tuple[List[Dict], int]:
    seed_by_id = {normalize_arxiv_id(seed["id"]): seed for seed in seeds}
    ids = list(seed_by_id.keys())
    fallback_count = 0

    try:
        results = list(client.results(arxiv.Search(id_list=ids, max_results=len(ids))))
    except Exception as exc:
        print(f"Metadata batch failed for {len(ids)} papers: {exc}", file=sys.stderr)
        return [fallback_metadata_item(seed, exc) for seed in seeds], len(seeds)

    result_by_id = {paper_id_from_result(result): result for result in results}
    enriched = []
    for seed in seeds:
        paper_id = normalize_arxiv_id(seed["id"])
        paper = result_by_id.get(paper_id)
        if paper is None:
            fallback_count += 1
            enriched.append(fallback_metadata_item(seed))
            continue

        categories = ordered_target_categories(seed.get("categories", []), target_categories)
        if not categories and paper.categories and paper.categories[0] in target_categories:
            categories = [paper.categories[0]]

        enriched.append({
            "id": paper_id,
            "categories": categories,
            "pdf": f"https://arxiv.org/pdf/{paper_id}",
            "abs": f"https://arxiv.org/abs/{paper_id}",
            "authors": [author.name for author in paper.authors],
            "title": paper.title,
            "comment": paper.comment,
            "summary": paper.summary,
        })

    return enriched, fallback_count


def write_jsonl(path: Path, rows: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stream_crawl_and_enhance(args) -> int:
    args.ai_workers = max(args.ai_workers, 1)
    args.metadata_batch_size = max(args.metadata_batch_size, 1)

    categories = parse_categories(args.categories)
    if not categories:
        raise ValueError("No arXiv categories configured")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"{args.date}.jsonl"
    enhanced_path = output_dir / f"{args.date}_AI_enhanced_{args.language}.jsonl"
    raw_tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
    enhanced_tmp = enhanced_path.with_suffix(enhanced_path.suffix + ".tmp")

    for path in (raw_tmp, enhanced_tmp):
        if path.exists():
            path.unlink()

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
        seed["categories"] = ordered_target_categories(seed.get("categories", []), categories)
        unique_seeds.append(seed)

    print(
        f"Candidates={len(seeds)}, new_after_dedup={len(unique_seeds)}, duplicates={duplicate_count}",
        file=sys.stderr,
    )
    if not unique_seeds:
        return 1

    work_queue: Queue = Queue(maxsize=args.ai_workers * 4)
    results = []
    results_lock = Lock()
    stop_token = object()
    chain = enhance.build_chain(args.model_name)

    print(
        f"Streaming AI enhancement with workers={args.ai_workers}, model={args.model_name}, "
        f"metadata_batch_size={args.metadata_batch_size}",
        file=sys.stderr,
    )

    def ai_worker(worker_index: int):
        while True:
            task = work_queue.get()
            try:
                if task is stop_token:
                    return
                seq, item = task
                processed = enhance.process_single_item(chain, item, args.language)
                with results_lock:
                    results.append((seq, processed))
            except Exception as exc:
                seq, item = task
                print(f"Worker {worker_index} failed for {item.get('id', 'unknown')}: {exc}", file=sys.stderr)
                enhance.apply_ai_fallback(
                    item,
                    "stream_worker_error",
                    exc=exc,
                    deferred=True,
                )
                with results_lock:
                    results.append((seq, item))
            finally:
                work_queue.task_done()

    workers = [
        Thread(target=ai_worker, args=(idx,), daemon=True)
        for idx in range(args.ai_workers)
    ]
    for worker in workers:
        worker.start()

    raw_rows = []
    metadata_fallback_count = 0
    client = arxiv.Client(
        page_size=args.metadata_batch_size,
        delay_seconds=args.metadata_delay_seconds,
        num_retries=args.metadata_retries,
    )

    seq = 0
    try:
        for batch in batched(unique_seeds, args.metadata_batch_size):
            metadata_items, fallback_count = enrich_metadata_batch(client, batch, categories)
            metadata_fallback_count += fallback_count
            for item in metadata_items:
                raw_rows.append(item)
                work_queue.put((seq, copy.deepcopy(item)))
                seq += 1

        work_queue.join()
    finally:
        for _ in workers:
            work_queue.put(stop_token)
        for worker in workers:
            worker.join(timeout=10)

    enhanced_rows = [
        item
        for _, item in sorted(results, key=lambda pair: pair[0])
        if item is not None
    ]

    if not raw_rows:
        return 1
    if not enhanced_rows:
        raise RuntimeError("AI enhancement produced no rows")

    write_jsonl(raw_tmp, raw_rows)
    write_jsonl(enhanced_tmp, enhanced_rows)
    raw_tmp.replace(raw_path)
    enhanced_tmp.replace(enhanced_path)

    print(
        f"Wrote raw={len(raw_rows)} enhanced={len(enhanced_rows)} "
        f"metadata_fallback={metadata_fallback_count}",
        file=sys.stderr,
    )
    return 0


def main():
    args = parse_args()
    try:
        exit_code = stream_crawl_and_enhance(args)
    except Exception as exc:
        print(f"Stream crawl/enhance failed: {exc}", file=sys.stderr)
        sys.exit(2)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
