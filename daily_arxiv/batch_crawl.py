#!/usr/bin/env python3
import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
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
    ordered_target_categories,
    parse_list_page,
    parse_categories,
    write_jsonl,
)


EXIT_NO_NEW_CONTENT = 1
EXIT_PROCESSING_ERROR = 2
EXIT_DEFERRED = 3
EXIT_LOW_CAPTURE = 4
ARXIV_ABS_URL_TEMPLATE = "https://arxiv.org/abs/{paper_id}"
ARXIV_LIST_URL_TEMPLATE = "https://arxiv.org/list/{category}/new"
ARXIV_RECENT_URL_TEMPLATE = "https://arxiv.org/list/{category}/recent?skip=0&show=2000"
WEEKDAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTH_LABELS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
MONTH_NAMES = {number: name for name, number in MONTH_LABELS.items()}
RECENT_SECTION_PATTERN = re.compile(
    r"<h3>(?P<label>[A-Za-z]{3}, \d{1,2} [A-Za-z]{3} \d{4})\s*"
    r"\(showing \d+ of (?P<total>\d+) entries"
)


class CrawlDeferred(RuntimeError):
    pass


class LowCaptureDeferred(RuntimeError):
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
    parser.add_argument("--low-capture-zero-fraction", type=float, default=env_float("ARXIV_LOW_CAPTURE_ZERO_FRACTION", 0.5))
    parser.add_argument("--low-capture-fraction", type=float, default=env_float("ARXIV_LOW_CAPTURE_FRACTION", 0.4))
    parser.add_argument("--low-capture-max-attempts", type=int, default=env_int("ARXIV_LOW_CAPTURE_MAX_ATTEMPTS", 3))
    return parser.parse_args()


def retry_delay(attempt: int, base_seconds: float) -> float:
    return min(base_seconds * (2 ** (attempt - 1)), 180.0)


def pending_seed_path(output_dir: Path, target_date: str) -> Path:
    return output_dir / "_pending" / f"{target_date}.jsonl"


def pending_metadata_path(output_dir: Path, target_date: str) -> Path:
    return output_dir / "_pending" / f"{target_date}.metadata.jsonl"


def low_capture_marker_path(output_dir: Path, target_date: str) -> Path:
    return output_dir / "_pending" / f"{target_date}.low-capture.json"


def read_low_capture_attempts(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("attempts", 0))
    except (json.JSONDecodeError, OSError, ValueError):
        return 0


def write_low_capture_marker(path: Path, attempts: int, candidate_count: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"attempts": attempts, "last_count": candidate_count}),
        encoding="utf-8",
    )


def clear_low_capture_marker(path: Path):
    if path.exists():
        path.unlink()
        print(f"Cleared low-capture marker: {path}", file=sys.stderr)


def history_raw_counts(output_dir: Path, target_date: str, history_days: int) -> List[int]:
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    counts = []
    for offset in range(1, history_days + 1):
        date_str = (target_dt - timedelta(days=offset)).strftime("%Y-%m-%d")
        path = output_dir / f"{date_str}.jsonl"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as stream:
            count = sum(1 for line in stream if line.strip())
        if count:
            counts.append(count)
    return counts


def is_low_capture(
    category_counts: List[int],
    history_counts: List[int],
    *,
    zero_fraction_threshold: float,
    total_fraction_threshold: float,
) -> bool:
    """Detect a crawl that hit stale or partially-announced /new pages.

    Real failure signature (e.g. 2026-10-15): most categories return zero
    parseable entries while a few already refreshed, so the day's capture is
    a small fraction of the recent norm. Zero total means no announcement
    happened at all and is handled by the no-new-content exit path.
    """
    if not category_counts:
        return False
    total = sum(category_counts)
    if total <= 0:
        return False
    zero_fraction = (
        sum(1 for count in category_counts if count <= 0) / len(category_counts)
    )
    if zero_fraction >= zero_fraction_threshold:
        return True
    if history_counts:
        median = statistics.median(history_counts)
        if median > 0 and total < total_fraction_threshold * median:
            return True
    return False


def recent_section_label(target_date: str) -> str:
    parsed = datetime.strptime(target_date, "%Y-%m-%d")
    return (
        f"{WEEKDAY_LABELS[parsed.weekday()]}, "
        f"{parsed.day} {MONTH_NAMES[parsed.month]} {parsed.year}"
    )


def parse_recent_section_date(label: str) -> str | None:
    match = re.fullmatch(r"([A-Za-z]{3}), (\d{1,2}) ([A-Za-z]{3}) (\d{4})", label)
    if match is None:
        return None
    month = MONTH_LABELS.get(match.group(3))
    if month is None:
        return None
    try:
        return datetime(int(match.group(4)), month, int(match.group(2))).strftime("%Y-%m-%d")
    except ValueError:
        return None


CROSS_LIST_SOURCE_PATTERN = re.compile(r"cross-list\s+from\s+([A-Za-z][\w.\-]*)")


def parse_recent_page(
    html: str,
    target_label: str,
    target_categories: set | None = None,
) -> tuple[int | None, str | None, int]:
    """Extract (target-date section total, top section date, non-target cross-lists).

    Section headers look like '<h3>Fri, 16 Oct 2026 (showing 22 of 22 entries )'
    and are ordered newest first; the target total is None when the target date
    has no section on the page. The third value counts entries in the target
    section annotated '(cross-list from X)' where X is not one of
    ``target_categories``: /new parsing never yields such entries as candidates
    (their primary category is foreign and the parser keeps only primary
    categories from the target set), so a section consisting solely of them is
    consistent with a zero /new parse and must not trigger rule A.
    """
    top_date = None
    target_total = None
    excluded = 0
    matches = list(RECENT_SECTION_PATTERN.finditer(html))
    for index, match in enumerate(matches):
        label = match.group("label")
        if top_date is None:
            top_date = parse_recent_section_date(label)
        if label != target_label:
            continue
        target_total = int(match.group("total"))
        if target_categories is not None:
            section_end = (
                matches[index + 1].start() if index + 1 < len(matches) else len(html)
            )
            section = html[match.end():section_end]
            excluded = sum(
                1
                for source in CROSS_LIST_SOURCE_PATTERN.findall(section)
                if source not in target_categories
            )
        break
    return target_total, top_date, excluded


def fetch_recent_report(
    categories: List[str],
    target_date: str,
    list_timeout: float,
    started_at: float,
    max_seconds: float,
) -> tuple[Dict[str, int | None], Dict[str, str | None], Dict[str, int]]:
    """Fetch each category's /recent page and locate the target-date section.

    Returns per-category maps of (section total or None when absent, top
    section date, non-target cross-list count in the target section). A
    category whose /recent page cannot be fetched is omitted from the maps so
    the caller treats it as unverifiable rather than inconsistent. Bounded
    retries: this verification must never hang a crawl.
    """
    target_label = recent_section_label(target_date)
    targets = set(categories)
    session = requests.Session()
    totals: Dict[str, int | None] = {}
    top_dates: Dict[str, str | None] = {}
    excluded_crosslists: Dict[str, int] = {}
    for category in categories:
        url = ARXIV_RECENT_URL_TEMPLATE.format(category=category)
        print(f"Fetching arXiv recent list: {url}", file=sys.stderr)
        attempt = 1
        while attempt <= 3:
            ensure_metadata_budget(started_at, max_seconds)
            try:
                response = http_get_with_timeout(
                    session,
                    url,
                    timeout_with_budget(list_timeout, started_at, max_seconds),
                )
                total, top_date, excluded = parse_recent_page(
                    response.text, target_label, targets
                )
                totals[category] = total
                top_dates[category] = top_date
                excluded_crosslists[category] = excluded
                break
            except Exception as exc:  # noqa: BLE001
                if not enhance.is_retryable_exception(exc) or attempt >= 3:
                    print(
                        f"Recent list check unavailable for {category}: "
                        f"{enhance.short_error(exc)}",
                        file=sys.stderr,
                    )
                    break
                delay = retry_delay_for_exception(attempt, 1.0, exc)
                sleep_with_budget(delay, started_at, max_seconds)
                attempt += 1
    return totals, top_dates, excluded_crosslists


def capture_inconsistencies(
    categories: List[str],
    category_counts: List[int],
    recent_totals: Dict[str, int | None],
    recent_top_dates: Dict[str, str | None],
    target_date: str,
    excluded_crosslists: Dict[str, int] | None = None,
) -> List[str]:
    """Cross-check /new parse results against the dated sections on /recent.

    Steady state (verified): every category with a same-day /recent section
    also yields candidates from /new, and /recent's top section matches the
    target date. The three failure signatures below mean /new is serving
    stale, partial, or newer-than-target content and must not be locked in.

    ``excluded_crosslists`` carries the per-category count of target-section
    entries that are cross-lists from categories outside the target set; /new
    parsing never yields those as candidates, so they are subtracted before
    the parsed-zero comparison (2026-10-27: cs.OS announced only two
    cross-lists from cs.CR/cs.AR and /new's correct zero was flagged as a
    stale capture, blocking the whole day).
    """
    issues = []
    for category, parsed in zip(categories, category_counts):
        if category not in recent_totals:
            continue
        total = recent_totals[category]
        top_date = recent_top_dates.get(category)
        excluded = (excluded_crosslists or {}).get(category, 0)
        effective_total = total - excluded if total is not None else None
        if effective_total is not None and effective_total > 0 and parsed <= 0:
            excluded_note = (
                f" ({excluded} non-target cross-lists excluded)" if excluded else ""
            )
            issues.append(
                f"{category}: /recent shows {total} entries for {target_date} "
                "but /new parsed 0 candidates"
                f"{excluded_note}"
            )
        elif parsed > 0 and total is None:
            issues.append(
                f"{category}: /new parsed {parsed} candidates but /recent has "
                f"no section for {target_date}"
            )
        if parsed > 0 and top_date is not None and top_date != target_date:
            issues.append(
                f"{category}: /recent top section is {top_date}, not {target_date} "
                "(/new is serving a different day's listing)"
            )
    return issues


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


def load_pending_metadata(path: Path, seed_ids: set, target_categories: List[str]) -> Dict[str, Dict]:
    metadata_by_id = {}
    for row in read_jsonl(path):
        paper_id = normalize_arxiv_id(row.get("id", ""))
        if paper_id in seed_ids and row.get("title") and row.get("summary"):
            row["id"] = paper_id
            row["categories"] = ordered_target_categories(row.get("categories", []), target_categories)
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
) -> tuple[List[Dict], List[int]]:
    session = requests.Session()
    seeds = []
    category_counts = []

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
                category_seeds = parse_list_page(response.text, categories)
                category_counts.append(len(category_seeds))
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

    return seeds, category_counts


def parse_abs_page_metadata(seed: Dict, html: str, target_categories: List[str]) -> Dict:
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
    categories = ordered_target_categories(seed.get("categories", []), target_categories)
    if not categories:
        primary_subjects_text = clean_text(" ".join(selector.css(".primary-subject::text").getall()))
        primary_categories = re.findall(r"\(([^)]+)\)", primary_subjects_text)
        if primary_categories and primary_categories[0] in target_categories:
            categories = [primary_categories[0]]
        elif not primary_categories:
            categories = ordered_target_categories(re.findall(r"\(([^)]+)\)", subjects_text), target_categories)

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
    target_categories: List[str],
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
        enriched.append(parse_abs_page_metadata(seed, response.text, target_categories))

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
    target_categories: List[str],
) -> List[Dict]:
    attempt = 1
    while True:
        ensure_metadata_budget(started_at, max_seconds)
        try:
            return metadata_from_batch(
                session,
                seeds,
                timeout,
                limiter,
                started_at,
                max_seconds,
                target_categories,
            )
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
                    target_categories,
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
                    target_categories,
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
    history_counts = history_raw_counts(output_dir, args.date, args.history_days)

    if pending_path.exists():
        unique_seeds = read_jsonl(pending_path)
        for seed in unique_seeds:
            seed["categories"] = ordered_target_categories(seed.get("categories", []), categories)
        print(f"Loaded pending seeds={len(unique_seeds)} from {pending_path}", file=sys.stderr)
        category_counts = []
    else:
        try:
            seeds, category_counts = crawl_seed_papers_with_budget(
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
            seed["categories"] = ordered_target_categories(seed.get("categories", []), categories)
            unique_seeds.append(seed)

        print(
            f"Candidates={len(seeds)}, new_after_dedup={len(unique_seeds)}, duplicates={duplicate_count}",
            file=sys.stderr,
        )

        marker_path = low_capture_marker_path(output_dir, args.date)
        if unique_seeds:
            recent_totals, recent_top_dates, recent_excluded = fetch_recent_report(
                categories,
                args.date,
                args.list_timeout,
                started_at,
                args.metadata_max_seconds,
            )
            issues = capture_inconsistencies(
                categories,
                category_counts,
                recent_totals,
                recent_top_dates,
                args.date,
                recent_excluded,
            )
            if issues:
                print(
                    f"Recent-listing verification found {len(issues)} "
                    "inconsistent categories",
                    file=sys.stderr,
                )
            heuristic = is_low_capture(
                category_counts,
                history_counts,
                zero_fraction_threshold=args.low_capture_zero_fraction,
                total_fraction_threshold=args.low_capture_fraction,
            )
            if issues or heuristic:
                attempts = read_low_capture_attempts(marker_path) + 1
                median = statistics.median(history_counts) if history_counts else 0
                if attempts < args.low_capture_max_attempts:
                    if issues:
                        reason = "; ".join(issues[:3])
                        if len(issues) > 3:
                            reason += f"; (+{len(issues) - 3} more categories)"
                    else:
                        reason = (
                            f"suspiciously low capture: {len(unique_seeds)} papers from "
                            f"{sum(1 for count in category_counts if count > 0)}/"
                            f"{len(category_counts)} categories vs history median {median:.0f}"
                        )
                    write_low_capture_marker(marker_path, attempts, len(unique_seeds))
                    raise LowCaptureDeferred(
                        f"/new vs /recent mismatch: {reason}; raw not written, "
                        f"next scheduled run will re-crawl "
                        f"(attempt {attempts}/{args.low_capture_max_attempts})"
                    )
                print(
                    f"Capture verification has not recovered after {attempts} attempts; "
                    "accepting the capture as the real listing",
                    file=sys.stderr,
                )
            clear_low_capture_marker(marker_path)
            checkpoint_pending_seeds(pending_path, unique_seeds)

    if not unique_seeds:
        clear_pending_file(pending_path, "seed")
        clear_pending_file(metadata_pending_path, "metadata")
        return EXIT_NO_NEW_CONTENT

    seed_ids = {normalize_arxiv_id(seed["id"]) for seed in unique_seeds}
    metadata_by_id = load_pending_metadata(metadata_pending_path, seed_ids, categories)
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
            categories,
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
    except LowCaptureDeferred as exc:
        print(f"Batch crawl low-capture defer: {exc}", file=sys.stderr)
        sys.exit(EXIT_LOW_CAPTURE)
    except CrawlDeferred as exc:
        print(f"Batch crawl deferred: {exc}", file=sys.stderr)
        sys.exit(EXIT_DEFERRED)
    except Exception as exc:
        print(f"Batch crawl failed: {exc}", file=sys.stderr)
        sys.exit(EXIT_PROCESSING_ERROR)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
