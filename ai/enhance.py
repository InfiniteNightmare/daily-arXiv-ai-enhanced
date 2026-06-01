import os
import json
import sys
import re
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from threading import Lock
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from pathlib import Path

import requests
from scrapy import Selector

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

AI_DIR = Path(__file__).resolve().parent
if (AI_DIR / ".env").exists():
    dotenv.load_dotenv(AI_DIR / ".env")
template = (AI_DIR / "template.txt").read_text()
system = (AI_DIR / "system.txt").read_text()

RETRYABLE_HTTP_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 522, 524}
REQUIRED_AI_FIELDS = ["tldr", "motivation", "method", "result", "conclusion"]
SENSITIVE_CHECK_URL = os.environ.get("SENSITIVE_CHECK_URL", "https://spam.dw-dengwei.workers.dev")


def get_env_int(name: str, default: int, min_value: int = 0) -> int:
    try:
        value = int(os.environ.get(name, default))
        return max(value, min_value)
    except ValueError:
        print(f"Invalid integer for {name}; using {default}", file=sys.stderr)
        return default


def get_env_float(name: str, default: float, min_value: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, default))
        return max(value, min_value)
    except ValueError:
        print(f"Invalid float for {name}; using {default}", file=sys.stderr)
        return default


def get_env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


AI_RETRY_ATTEMPTS = get_env_int("AI_RETRY_ATTEMPTS", 3, min_value=1)
AI_RETRY_BASE_SECONDS = get_env_float("AI_RETRY_BASE_SECONDS", 2.0)
AI_RETRY_MAX_SECONDS = get_env_float("AI_RETRY_MAX_SECONDS", 60.0)
AI_REQUEST_TIMEOUT_SECONDS = get_env_float("AI_REQUEST_TIMEOUT_SECONDS", 300.0, min_value=1.0)
AI_SDK_MAX_RETRIES = get_env_int("AI_SDK_MAX_RETRIES", 0, min_value=0)
AI_CIRCUIT_BREAKER_FAILURES = get_env_int("AI_CIRCUIT_BREAKER_FAILURES", 0, min_value=0)
AI_MIN_INTERVAL_SECONDS = get_env_float("AI_MIN_INTERVAL_SECONDS", 0.0)
AI_MAX_SECONDS = get_env_float("AI_MAX_SECONDS", 0.0)
AI_INPUT_SOURCE = os.environ.get("AI_INPUT_SOURCE", "html").strip().lower()
AI_HTML_TIMEOUT_SECONDS = get_env_float("AI_HTML_TIMEOUT_SECONDS", 20.0, min_value=1.0)
AI_HTML_RETRY_ATTEMPTS = get_env_int("AI_HTML_RETRY_ATTEMPTS", 2, min_value=1)
AI_HTML_RETRY_BASE_SECONDS = get_env_float("AI_HTML_RETRY_BASE_SECONDS", 2.0)
AI_HTML_RETRY_MAX_SECONDS = get_env_float("AI_HTML_RETRY_MAX_SECONDS", 30.0)
AI_HTML_MIN_INTERVAL_SECONDS = get_env_float("AI_HTML_MIN_INTERVAL_SECONDS", 0.5)
AI_HTML_MAX_CHARS = get_env_int("AI_HTML_MAX_CHARS", 0, min_value=0)
AI_HTML_MIN_CHARS = get_env_int("AI_HTML_MIN_CHARS", 2000, min_value=0)

SENSITIVE_CHECK_ENABLED = get_env_bool("SENSITIVE_CHECK_ENABLED", False)
SENSITIVE_CHECK_RETRY_ATTEMPTS = get_env_int("SENSITIVE_CHECK_RETRY_ATTEMPTS", 2, min_value=1)
SENSITIVE_CHECK_RETRY_BASE_SECONDS = get_env_float("SENSITIVE_CHECK_RETRY_BASE_SECONDS", 1.0)
SENSITIVE_CHECK_RETRY_MAX_SECONDS = get_env_float("SENSITIVE_CHECK_RETRY_MAX_SECONDS", 15.0)
SENSITIVE_CHECK_CIRCUIT_BREAKER_FAILURES = get_env_int("SENSITIVE_CHECK_CIRCUIT_BREAKER_FAILURES", 5, min_value=1)

STATE_LOCK = Lock()
AI_RATE_LIMIT_LOCK = Lock()
HTML_RATE_LIMIT_LOCK = Lock()
AI_CONSECUTIVE_FAILURES = 0
AI_CIRCUIT_OPEN = False
AI_LAST_REQUEST_AT = 0.0
HTML_LAST_REQUEST_AT = 0.0
SENSITIVE_CHECK_CONSECUTIVE_FAILURES = 0
SENSITIVE_CHECK_CIRCUIT_OPEN = False


def short_error(exc: Exception, limit: int = 500) -> str:
    message = re.sub(r"\s+", " ", str(exc)).strip()
    return message[:limit]


def get_status_code(exc: Exception):
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)

    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


def is_retryable_exception(exc: Exception) -> bool:
    status_code = get_status_code(exc)
    if status_code is None:
        return True
    return status_code in RETRYABLE_HTTP_STATUS_CODES or status_code >= 500


def parse_retry_after_seconds(exc: Exception):
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) if response is not None else {}
    retry_after = None
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if not retry_after:
        return None

    try:
        return max(float(retry_after), 0.0)
    except ValueError:
        pass

    try:
        retry_dt = parsedate_to_datetime(retry_after)
        if retry_dt.tzinfo is None:
            retry_dt = retry_dt.replace(tzinfo=timezone.utc)
        return max((retry_dt - datetime.now(timezone.utc)).total_seconds(), 0.0)
    except (TypeError, ValueError):
        return None


def retry_delay_seconds(attempt: int, base_seconds: float, max_seconds: float, exc: Exception) -> float:
    retry_after = parse_retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, max_seconds)

    exponential_delay = min(base_seconds * (2 ** (attempt - 1)), max_seconds)
    jitter = random.uniform(0, min(base_seconds, exponential_delay * 0.25))
    return min(exponential_delay + jitter, max_seconds)


def build_fallback_ai_fields(item: Dict, reason: str, exc: Exception = None) -> Dict:
    abstract = item.get("summary") or "Original abstract unavailable."
    details = f"AI enhancement failed after retries ({reason}). Original arXiv metadata is preserved."
    if exc is not None:
        details = f"{details} Error: {short_error(exc)}"

    return {
        "tldr": abstract,
        "motivation": details,
        "method": "AI enhancement unavailable; see the original abstract.",
        "result": "AI enhancement unavailable; see the original abstract.",
        "conclusion": details,
    }


def apply_ai_fallback(item: Dict, reason: str, error: str = None, exc: Exception = None) -> Dict:
    item['AI'] = build_fallback_ai_fields(item, reason, exc)
    item['AI_status'] = "fallback"
    if error is not None:
        item['AI_error'] = error
    elif exc is not None:
        item['AI_error'] = short_error(exc)
    return item


def has_metadata_fallback(item: Dict) -> bool:
    if item.get("metadata_status") == "fallback":
        return True
    summary = item.get("summary") or ""
    return summary.startswith("arXiv metadata fetch failed after retries.")


def is_ai_circuit_open() -> bool:
    with STATE_LOCK:
        return AI_CIRCUIT_OPEN


def record_ai_success():
    global AI_CONSECUTIVE_FAILURES
    with STATE_LOCK:
        AI_CONSECUTIVE_FAILURES = 0


def record_ai_failure(item_id: str, exc: Exception):
    global AI_CONSECUTIVE_FAILURES, AI_CIRCUIT_OPEN
    with STATE_LOCK:
        AI_CONSECUTIVE_FAILURES += 1
        if (
            AI_CIRCUIT_BREAKER_FAILURES > 0
            and not AI_CIRCUIT_OPEN
            and AI_CONSECUTIVE_FAILURES >= AI_CIRCUIT_BREAKER_FAILURES
        ):
            AI_CIRCUIT_OPEN = True
            print(
                f"AI circuit opened after {AI_CONSECUTIVE_FAILURES} consecutive failures; "
                f"remaining items will keep metadata with fallback AI fields. Last item: {item_id}. "
                f"Last error: {short_error(exc)}",
                file=sys.stderr,
            )


def wait_for_ai_rate_limit():
    global AI_LAST_REQUEST_AT
    if AI_MIN_INTERVAL_SECONDS <= 0:
        return

    with AI_RATE_LIMIT_LOCK:
        now = time.monotonic()
        wait_seconds = AI_LAST_REQUEST_AT + AI_MIN_INTERVAL_SECONDS - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now = time.monotonic()
        AI_LAST_REQUEST_AT = now


def is_ai_time_budget_exceeded(started_at: float) -> bool:
    return AI_MAX_SECONDS > 0 and (time.monotonic() - started_at) >= AI_MAX_SECONDS


def wait_for_html_rate_limit():
    global HTML_LAST_REQUEST_AT
    if AI_HTML_MIN_INTERVAL_SECONDS <= 0:
        return

    with HTML_RATE_LIMIT_LOCK:
        now = time.monotonic()
        wait_seconds = HTML_LAST_REQUEST_AT + AI_HTML_MIN_INTERVAL_SECONDS - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now = time.monotonic()
        HTML_LAST_REQUEST_AT = now


def truncate_for_prompt(content: str, max_chars: int) -> str:
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if max_chars <= 0:
        return content
    if len(content) <= max_chars:
        return content

    head_chars = int(max_chars * 0.7)
    tail_chars = max_chars - head_chars
    return (
        content[:head_chars].rstrip()
        + "\n\n[Content truncated for model context]\n\n"
        + content[-tail_chars:].lstrip()
    )


def extract_html_text(html: str) -> str:
    selector = Selector(text=html)
    for node in selector.xpath(
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_bibliography ')]"
        " | //section[.//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6]"
        "[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'references') "
        "or contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'bibliography')]]"
    ):
        root = getattr(node, "root", None)
        if root is not None:
            parent = root.getparent()
            if parent is not None:
                parent.remove(root)

    text_nodes = selector.xpath(
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_document ') "
        "or contains(concat(' ', normalize-space(@class), ' '), ' ltx_page_content ') "
        "or self::main or self::article]"
        "//text()[not(ancestor::script) and not(ancestor::style) and not(ancestor::nav) "
        "and not(ancestor::header) and not(ancestor::footer)]"
    ).getall()
    if not text_nodes:
        text_nodes = selector.xpath(
            "//body//text()[not(ancestor::script) and not(ancestor::style) "
            "and not(ancestor::nav) and not(ancestor::header) and not(ancestor::footer)]"
        ).getall()

    lines = []
    for node in text_nodes:
        text = re.sub(r"\s+", " ", node).strip()
        if text:
            lines.append(text)
    text = "\n".join(lines)
    text = re.split(
        r"\n\s*(references|bibliography)\s*\n",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return text.strip()


def fetch_html_text(item: Dict):
    if AI_INPUT_SOURCE not in {"html", "fulltext"}:
        return None

    paper_id = item.get("id")
    if not paper_id:
        return None

    url = f"https://arxiv.org/html/{paper_id}"
    last_error = None
    for attempt in range(1, AI_HTML_RETRY_ATTEMPTS + 1):
        try:
            wait_for_html_rate_limit()
            response = requests.get(url, timeout=AI_HTML_TIMEOUT_SECONDS)
            if response.status_code == 404:
                return None
            if response.status_code != 200:
                http_error = requests.HTTPError(f"HTML fetch failed with status {response.status_code}")
                http_error.response = response
                raise http_error

            fulltext = extract_html_text(response.text)
            if len(fulltext) < AI_HTML_MIN_CHARS:
                raise ValueError(f"HTML text too short: {len(fulltext)} chars")

            return truncate_for_prompt(fulltext, AI_HTML_MAX_CHARS)
        except Exception as exc:
            last_error = exc
            if attempt >= AI_HTML_RETRY_ATTEMPTS or not is_retryable_exception(exc):
                print(
                    f"HTML unavailable for {paper_id}; using abstract: {short_error(exc)}",
                    file=sys.stderr,
                )
                return None

            delay = retry_delay_seconds(
                attempt,
                AI_HTML_RETRY_BASE_SECONDS,
                AI_HTML_RETRY_MAX_SECONDS,
                exc,
            )
            print(
                f"HTML retry {attempt}/{AI_HTML_RETRY_ATTEMPTS - 1} for {paper_id} "
                f"after {delay:.1f}s: {short_error(exc)}",
                file=sys.stderr,
            )
            time.sleep(delay)

    print(f"HTML unavailable for {paper_id}; using abstract: {short_error(last_error)}", file=sys.stderr)
    return None


def build_ai_input(item: Dict) -> str:
    abstract = item.get("summary", "")
    html_text = fetch_html_text(item)
    if html_text:
        item["AI_input_source"] = "arxiv_html"
        item["AI_input_chars"] = len(html_text)
        return (
            f"Title: {item.get('title', '')}\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Paper full text from arXiv HTML:\n{html_text}"
        )

    item["AI_input_source"] = "abstract"
    item["AI_input_chars"] = len(abstract)
    return abstract


def invoke_with_retries(chain, payload: Dict, item_id: str) -> Structure:
    for attempt in range(1, AI_RETRY_ATTEMPTS + 1):
        try:
            wait_for_ai_rate_limit()
            response = chain.invoke(payload)
            if response is None:
                raise ValueError("AI returned empty structured response")
            if not hasattr(response, "model_dump"):
                raise ValueError(f"AI returned unsupported structured response: {type(response).__name__}")
            return response
        except Exception as exc:
            if attempt >= AI_RETRY_ATTEMPTS or not is_retryable_exception(exc):
                raise

            delay = retry_delay_seconds(attempt, AI_RETRY_BASE_SECONDS, AI_RETRY_MAX_SECONDS, exc)
            print(
                f"AI retry {attempt}/{AI_RETRY_ATTEMPTS - 1} for {item_id} "
                f"after {delay:.1f}s: {short_error(exc)}",
                file=sys.stderr,
            )
            time.sleep(delay)


def is_sensitive_check_circuit_open() -> bool:
    with STATE_LOCK:
        return SENSITIVE_CHECK_CIRCUIT_OPEN


def record_sensitive_check_success():
    global SENSITIVE_CHECK_CONSECUTIVE_FAILURES
    with STATE_LOCK:
        SENSITIVE_CHECK_CONSECUTIVE_FAILURES = 0


def record_sensitive_check_failure(exc: Exception):
    global SENSITIVE_CHECK_CONSECUTIVE_FAILURES, SENSITIVE_CHECK_CIRCUIT_OPEN
    with STATE_LOCK:
        SENSITIVE_CHECK_CONSECUTIVE_FAILURES += 1
        if (
            not SENSITIVE_CHECK_CIRCUIT_OPEN
            and SENSITIVE_CHECK_CONSECUTIVE_FAILURES >= SENSITIVE_CHECK_CIRCUIT_BREAKER_FAILURES
        ):
            SENSITIVE_CHECK_CIRCUIT_OPEN = True
            print(
                f"Sensitive check circuit opened after {SENSITIVE_CHECK_CONSECUTIVE_FAILURES} "
                f"consecutive failures; remaining checks will keep items. "
                f"Last error: {short_error(exc)}",
                file=sys.stderr,
            )


def is_sensitive(content: str) -> bool:
    """
    调用 spam.dw-dengwei.workers.dev 接口检测内容是否包含敏感词。
    返回 True 表示触发敏感词，False 表示未触发。
    检测服务失败时 fail open，避免临时 429/网络错误把整天数据过滤为空。
    """
    if not SENSITIVE_CHECK_ENABLED:
        return False

    if not content:
        return False

    if is_sensitive_check_circuit_open():
        return False

    last_error = None
    for attempt in range(1, SENSITIVE_CHECK_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(
                SENSITIVE_CHECK_URL,
                json={"text": content},
                timeout=5
            )
            if resp.status_code == 200:
                result = resp.json()
                record_sensitive_check_success()
                return bool(result.get("sensitive", False))

            http_error = requests.HTTPError(f"Sensitive check failed with status {resp.status_code}")
            http_error.response = resp
            if not is_retryable_exception(http_error):
                record_sensitive_check_failure(http_error)
                print(f"Sensitive check non-retryable status {resp.status_code}; keeping item", file=sys.stderr)
                return False
            raise http_error
        except Exception as exc:
            last_error = exc
            if attempt >= SENSITIVE_CHECK_RETRY_ATTEMPTS or not is_retryable_exception(exc):
                record_sensitive_check_failure(exc)
                print(f"Sensitive check unavailable; keeping item: {short_error(exc)}", file=sys.stderr)
                return False

            delay = retry_delay_seconds(
                attempt,
                SENSITIVE_CHECK_RETRY_BASE_SECONDS,
                SENSITIVE_CHECK_RETRY_MAX_SECONDS,
                exc,
            )
            print(
                f"Sensitive check retry {attempt}/{SENSITIVE_CHECK_RETRY_ATTEMPTS - 1} "
                f"after {delay:.1f}s: {short_error(exc)}",
                file=sys.stderr,
            )
            time.sleep(delay)

    record_sensitive_check_failure(last_error)
    print(f"Sensitive check unavailable; keeping item: {short_error(last_error)}", file=sys.stderr)
    return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()


def build_chain(model_name: str):
    llm = ChatOpenAI(
        model=model_name,
        timeout=AI_REQUEST_TIMEOUT_SECONDS,
        max_retries=AI_SDK_MAX_RETRIES,
    ).with_structured_output(Structure, method="function_calling")

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    return prompt_template | llm


def process_single_item(chain, item: Dict, language: str) -> Dict:
    def check_github_code(content: str) -> Dict:
        """提取并验证 GitHub 链接"""
        code_info = {}

        # 1. 优先匹配 github.com/owner/repo 格式
        github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
        match = re.search(github_pattern, content)
        
        if match:
            owner, repo = match.groups()
            # 清理 repo 名称，去掉可能的 .git 后缀或末尾的标点
            repo = repo.rstrip(".git").rstrip(".,)")
            
            full_url = f"https://github.com/{owner}/{repo}"
            code_info["code_url"] = full_url
            
            # 尝试调用 GitHub API 获取信息
            github_token = os.environ.get("TOKEN_GITHUB")
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            try:
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                resp = requests.get(api_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    code_info["code_stars"] = data.get("stargazers_count", 0)
                    code_info["code_last_update"] = data.get("pushed_at", "")[:10]
            except Exception:
                # API 调用失败不影响主流程
                pass
            return code_info

        # 2. 如果没有 github.com，尝试匹配 github.io
        github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
        match_io = re.search(github_io_pattern, content)
        
        if match_io:
            url = match_io.group(0)
            # 清理末尾标点
            url = url.rstrip(".,)")
            code_info["code_url"] = url
            # github.io 不进行 star 和 update 判断
                
        return code_info

    ai_input = build_ai_input(item)

    # 检查输入内容
    if is_sensitive(ai_input):
        return None

    # 检测代码可用性
    code_info = check_github_code(ai_input)
    if code_info:
        item.update(code_info)

    fallback_ai_fields = build_fallback_ai_fields(item, "not_started")

    if has_metadata_fallback(item):
        apply_ai_fallback(
            item,
            "metadata_unavailable",
            "AI enhancement skipped because arXiv metadata was unavailable.",
        )
    elif is_ai_circuit_open():
        apply_ai_fallback(
            item,
            "ai_circuit_open",
            "AI enhancement skipped after repeated failures in this run.",
        )
    else:
        try:
            response: Structure = invoke_with_retries(chain, {
                "language": language,
                "content": ai_input
            }, item.get("id", "unknown"))
            item['AI'] = response.model_dump()
            item['AI_status'] = "ok"
            item.pop('AI_error', None)
            record_ai_success()
        except langchain_core.exceptions.OutputParserException as e:
            record_ai_failure(item.get("id", "unknown"), e)
            # 尝试从错误信息中提取 JSON 字符串并修复
            error_msg = str(e)
            partial_data = {}

            if "Function Structure arguments:" in error_msg:
                try:
                    # 提取 JSON 字符串
                    json_str = error_msg.split("Function Structure arguments:", 1)[1].strip().split('are not valid JSON')[0].strip()
                    # 预处理 LaTeX 数学符号 - 使用四个反斜杠来确保正确转义
                    json_str = json_str.replace('\\', '\\\\')
                    # 尝试解析修复后的 JSON
                    partial_data = json.loads(json_str)
                except Exception as json_e:
                    print(f"Failed to parse JSON for {item.get('id', 'unknown')}: {json_e}", file=sys.stderr)

            # Merge partial data with fallback data to ensure metadata is preserved.
            item['AI'] = {**build_fallback_ai_fields(item, "output_parser_error", e), **partial_data}
            item['AI_status'] = "partial" if partial_data else "fallback"
            item['AI_error'] = short_error(e)
            print(f"Using partial AI data for {item.get('id', 'unknown')}: {list(partial_data.keys())}", file=sys.stderr)
        except Exception as e:
            record_ai_failure(item.get("id", "unknown"), e)
            # Catch any other exceptions and keep the original arXiv metadata.
            print(f"Unexpected error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
            apply_ai_fallback(item, "llm_error", exc=e)
    
    # Final validation to ensure all required fields exist
    for field in REQUIRED_AI_FIELDS:
        if field not in item['AI']:
            item['AI'][field] = fallback_ai_fields[field]

    # 检查 AI 生成的所有字段
    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            return None
    return item


def is_resumable_result(item: Dict) -> bool:
    if not item.get("id") or "AI" not in item:
        return False
    if AI_INPUT_SOURCE in {"html", "fulltext"} and "AI_input_source" not in item:
        return False
    return True


def load_existing_results(target_file: str, raw_ids: set) -> Dict:
    if not os.path.exists(target_file):
        return {}

    results = {}
    with open(target_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid checkpoint line: {short_error(exc)}", file=sys.stderr)
                continue
            item_id = item.get("id")
            if item_id in raw_ids and is_resumable_result(item):
                results[item_id] = item
    return results


def save_checkpoint(target_file: str, ordered_data: List[Dict], processed_by_id: Dict):
    target_path = Path(target_file)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    written = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for source_item in ordered_data:
            item = processed_by_id.get(source_item.get("id"))
            if item is not None:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
    tmp_path.replace(target_path)
    print(f"AI checkpoint saved: {written}/{len(ordered_data)} rows -> {target_file}", file=sys.stderr)


def ordered_processed_rows(data: List[Dict], processed_by_id: Dict) -> List[Dict]:
    return [
        processed_by_id[item["id"]]
        for item in data
        if item.get("id") in processed_by_id
    ]


def process_all_items(
    data: List[Dict],
    model_name: str,
    language: str,
    max_workers: int,
    target_file: str,
    processed_by_id: Dict,
) -> List[Dict]:
    """并行处理所有数据项"""
    print('Connect to:', model_name, file=sys.stderr)
    print(
        f"AI retry config: attempts={AI_RETRY_ATTEMPTS}, sdk_max_retries={AI_SDK_MAX_RETRIES}, "
        f"base={AI_RETRY_BASE_SECONDS}s, max={AI_RETRY_MAX_SECONDS}s, "
        f"time_budget={AI_MAX_SECONDS}s",
        file=sys.stderr,
    )

    chain = build_chain(model_name)
    pending_data = [item for item in data if item.get("id") not in processed_by_id]
    print(
        f"AI resume state: existing={len(processed_by_id)}, pending={len(pending_data)}, total={len(data)}",
        file=sys.stderr,
    )
    if not pending_data:
        save_checkpoint(target_file, data, processed_by_id)
        return ordered_processed_rows(data, processed_by_id)

    if AI_MAX_SECONDS > 0:
        started_at = time.monotonic()
        for item in tqdm(pending_data, total=len(pending_data), desc="Processing items"):
            item_id = item.get("id", "unknown")
            if is_ai_time_budget_exceeded(started_at):
                processed_by_id[item_id] = apply_ai_fallback(
                    item,
                    "time_budget_exceeded",
                    "AI enhancement skipped because the workflow time budget was reached.",
                )
                save_checkpoint(target_file, data, processed_by_id)
                continue
            try:
                result = process_single_item(chain, item, language)
                if result is None:
                    continue
                processed_by_id[item_id] = result
            except Exception as e:
                print(f"Item {item.get('id', 'unknown')} generated an exception: {e}", file=sys.stderr)
                processed_by_id[item_id] = apply_ai_fallback(item, "worker_error", exc=e)
            save_checkpoint(target_file, data, processed_by_id)
        return ordered_processed_rows(data, processed_by_id)
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language): idx
            for idx, item in enumerate(pending_data)
        }
        
        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(pending_data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            source_item = pending_data[idx]
            item_id = source_item.get("id", "unknown")
            try:
                result = future.result()
                if result is not None:
                    processed_by_id[item_id] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # Keep metadata even when worker-level processing fails.
                processed_by_id[item_id] = source_item
                processed_by_id[item_id]['AI'] = build_fallback_ai_fields(source_item, "worker_error", e)
                processed_by_id[item_id]['AI_status'] = "fallback"
                processed_by_id[item_id]['AI_error'] = short_error(e)
            save_checkpoint(target_file, data, processed_by_id)
    
    return ordered_processed_rows(data, processed_by_id)

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)
    existing_results = load_existing_results(target_file, seen_ids)
    
    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers,
        target_file,
        existing_results,
    )
    
    # 保存结果
    save_checkpoint(target_file, data, {item["id"]: item for item in processed_data})

if __name__ == "__main__":
    main()
