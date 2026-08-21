import copy
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "ai"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

import enhance  # noqa: E402


class DummyResponse:
    def __init__(self, status_code=None, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class ProviderError(Exception):
    def __init__(
        self,
        message,
        *,
        status_code=None,
        response=None,
        body=None,
    ):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if response is not None:
            self.response = response
        if body is not None:
            self.body = body


class FakeChain:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        raise AssertionError("The fake chain was not expected to be invoked")


def minimax_balance_error():
    body = {
        "type": "error",
        "error": {
            "type": "insufficient_balance_error",
            "message": (
                "当前已达到 Token Plan 用量上限。为避免调用中断，请升级 Token Plan 套餐，"
                "或购买积分补充用量并开启积分自动消耗。 (2067)"
            ),
            "http_code": "402",
        },
        "request_id": "06d3693a40fe9b4c0336166c3bb5dece",
    }
    return ProviderError(
        f"Error code: 402 - {body}",
        status_code=402,
        response=DummyResponse(402),
        body=body,
    )


def ordinary_422_error():
    body = {
        "type": "error",
        "error": {
            "type": "unprocessable_entity_error",
            "message": "input new_sensitive (1026)",
            "http_code": "422",
        },
        "request_id": "06d6f8df98a84f336161b1e2b34aa585",
    }
    return ProviderError(
        f"Error code: 422 - {body}",
        status_code=422,
        response=DummyResponse(422),
        body=body,
    )


def quota_429_error():
    body = {
        "error": {
            "type": "insufficient_quota",
            "code": "credit_balance_exhausted",
            "message": "The account has exhausted its available quota.",
        }
    }
    return ProviderError(
        f"Error code: 429 - {body}",
        status_code=429,
        response=DummyResponse(429),
        body=body,
    )


def ordinary_429_error():
    body = {
        "error": {
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
            "message": "Too many requests; retry later.",
        }
    }
    return ProviderError(
        f"Error code: 429 - {body}",
        status_code=429,
        response=DummyResponse(429),
        body=body,
    )


def sample_item():
    return {
        "id": "2608.12345",
        "title": "A Regression Test Paper",
        "summary": "The original arXiv abstract must be preserved.",
        "authors": ["Ada Lovelace", "Alan Turing"],
        "categories": ["cs.AI"],
        "pdf": "https://arxiv.org/pdf/2608.12345",
        "abs": "https://arxiv.org/abs/2608.12345",
        "comment": "Test metadata",
    }


def complete_ai_fields(details="generated"):
    return {
        "tldr": details,
        "motivation": details,
        "method": details,
        "result": details,
        "conclusion": details,
    }


class EnhanceStateTestCase(unittest.TestCase):
    _STATE_DEFAULTS = {
        "AI_CIRCUIT_OPEN": False,
        "AI_CONSECUTIVE_FAILURES": 0,
        "AI_CIRCUIT_BREAKER_FAILURES": 0,
        "AI_INPUT_SOURCE": "abstract",
        "AI_RETRY_ATTEMPTS": 3,
        "AI_MIN_INTERVAL_SECONDS": 0.0,
        "SENSITIVE_CHECK_ENABLED": False,
    }

    def setUp(self):
        self._saved_state = {
            name: getattr(enhance, name)
            for name in self._STATE_DEFAULTS
            if hasattr(enhance, name)
        }
        for name, value in self._STATE_DEFAULTS.items():
            if hasattr(enhance, name):
                setattr(enhance, name, value)

    def tearDown(self):
        for name, value in self._saved_state.items():
            setattr(enhance, name, value)

    def assert_metadata_preserved(self, original, processed):
        for key, value in original.items():
            self.assertEqual(
                processed.get(key),
                value,
                f"original metadata field {key!r} changed",
            )


class StatusCodeExtractionTests(EnhanceStateTestCase):
    def test_get_status_code_extracts_all_supported_exception_shapes(self):
        wrapped = RuntimeError("LangChain wrapper")
        wrapped.__cause__ = ProviderError("inner", status_code=402)

        cases = {
            "top_level": ProviderError("top-level", status_code="402"),
            "response": ProviderError(
                "response",
                response=DummyResponse(status_code="402"),
            ),
            "body": ProviderError("body", body={"http_code": "402"}),
            "nested_error": ProviderError(
                "nested",
                body={"type": "error", "error": {"http_code": "402"}},
            ),
            "cause": wrapped,
            "text": RuntimeError(
                "Error code: 402 - {'error': {'type': 'insufficient_balance_error'}}"
            ),
        }

        for name, error in cases.items():
            with self.subTest(shape=name):
                self.assertEqual(enhance.get_status_code(error), 402)

    def test_run_terminal_classification_distinguishes_account_and_paper_errors(self):
        terminal_errors = {
            "authentication": ProviderError("invalid key", status_code=401),
            "billing_status": ProviderError("billing", status_code=402),
            "balance_type": ProviderError(
                "balance",
                body={"type": "insufficient_balance_error"},
            ),
            "quota_nested": ProviderError(
                "quota",
                status_code=429,
                body={
                    "error": {
                        "type": "insufficient_quota",
                        "code": "credit_balance_exhausted",
                    }
                },
            ),
        }

        for name, error in terminal_errors.items():
            with self.subTest(error=name):
                self.assertTrue(enhance.is_run_terminal_ai_exception(error))

        self.assertFalse(enhance.is_run_terminal_ai_exception(ordinary_422_error()))


class CircuitBreakerTests(EnhanceStateTestCase):
    def test_account_error_opens_circuit_immediately_even_when_threshold_is_disabled(self):
        enhance.AI_CIRCUIT_BREAKER_FAILURES = 0

        enhance.record_ai_failure("2608.12345", minimax_balance_error())

        self.assertTrue(enhance.is_ai_circuit_open())

    def test_ordinary_422_does_not_open_circuit(self):
        enhance.record_ai_failure("2608.12345", ordinary_422_error())

        self.assertFalse(enhance.is_ai_circuit_open())


class RetryPolicyTests(EnhanceStateTestCase):
    def test_quota_429_stops_immediately_while_ordinary_429_exhausts_retries(self):
        quota_chain = FakeChain(quota_429_error())

        with mock.patch.object(enhance.time, "sleep") as quota_sleep:
            with self.assertRaises(ProviderError):
                enhance.invoke_with_retries(
                    quota_chain,
                    {"language": "Chinese", "content": "quota"},
                    "2608.quota",
                )

        self.assertEqual(len(quota_chain.calls), 1)
        quota_sleep.assert_not_called()

        rate_limit_chain = FakeChain(ordinary_429_error())
        with (
            mock.patch.object(enhance.time, "sleep") as rate_limit_sleep,
            mock.patch.object(enhance.random, "uniform", return_value=0.0),
        ):
            with self.assertRaises(ProviderError):
                enhance.invoke_with_retries(
                    rate_limit_chain,
                    {"language": "Chinese", "content": "rate limit"},
                    "2608.rate-limit",
                )

        self.assertEqual(len(rate_limit_chain.calls), enhance.AI_RETRY_ATTEMPTS)
        self.assertEqual(rate_limit_sleep.call_count, enhance.AI_RETRY_ATTEMPTS - 1)


class SingleItemFailureFlowTests(EnhanceStateTestCase):
    def test_open_circuit_skips_input_build_full_text_and_llm(self):
        original = sample_item()
        item = copy.deepcopy(original)
        chain = FakeChain()
        enhance.AI_CIRCUIT_OPEN = True

        with (
            mock.patch.object(
                enhance,
                "build_ai_input",
                side_effect=AssertionError("input must not be built after circuit opens"),
            ) as build_input,
            mock.patch.object(
                enhance,
                "fetch_full_text",
                side_effect=AssertionError("full text must not be fetched after circuit opens"),
            ) as fetch_full_text,
            mock.patch.object(enhance, "is_sensitive", return_value=False),
        ):
            processed = enhance.process_single_item(chain, item, "Chinese")

        build_input.assert_not_called()
        fetch_full_text.assert_not_called()
        self.assertEqual(chain.calls, [])
        self.assertEqual(processed["AI_status"], "deferred")
        self.assert_metadata_preserved(original, processed)
        self.assertEqual(set(enhance.REQUIRED_AI_FIELDS) - set(processed["AI"]), set())

    def test_402_defers_current_item_without_retry_wording(self):
        original = sample_item()
        item = copy.deepcopy(original)
        chain = FakeChain(minimax_balance_error())

        with (
            mock.patch.object(enhance, "build_ai_input", return_value=original["summary"]),
            mock.patch.object(enhance, "is_sensitive", return_value=False),
            mock.patch.object(enhance.time, "sleep") as sleep,
        ):
            processed = enhance.process_single_item(chain, item, "Chinese")

        self.assertEqual(len(chain.calls), 1)
        sleep.assert_not_called()
        self.assertTrue(enhance.is_ai_circuit_open())
        self.assertEqual(processed["AI_status"], "deferred")
        self.assertNotIn("after retries", json.dumps(processed["AI"]).lower())
        self.assert_metadata_preserved(original, processed)
        self.assertEqual(set(enhance.REQUIRED_AI_FIELDS) - set(processed["AI"]), set())

    def test_ordinary_422_falls_back_without_opening_circuit(self):
        original = sample_item()
        item = copy.deepcopy(original)
        chain = FakeChain(ordinary_422_error())

        with (
            mock.patch.object(enhance, "build_ai_input", return_value=original["summary"]),
            mock.patch.object(enhance, "is_sensitive", return_value=False),
            mock.patch.object(enhance.time, "sleep") as sleep,
        ):
            processed = enhance.process_single_item(chain, item, "Chinese")

        self.assertEqual(len(chain.calls), 1)
        sleep.assert_not_called()
        self.assertFalse(enhance.is_ai_circuit_open())
        self.assertEqual(processed["AI_status"], "fallback")
        self.assert_metadata_preserved(original, processed)
        self.assertEqual(set(enhance.REQUIRED_AI_FIELDS) - set(processed["AI"]), set())

    def test_first_402_defers_three_sequential_items_with_only_one_llm_call(self):
        items = []
        for suffix in range(1, 4):
            item = sample_item()
            item["id"] = f"2608.1234{suffix}"
            items.append(item)

        chain = FakeChain(minimax_balance_error())
        built_item_ids = []

        def build_input(item):
            built_item_ids.append(item["id"])
            return item["summary"]

        with (
            mock.patch.object(enhance, "build_ai_input", side_effect=build_input),
            mock.patch.object(enhance, "is_sensitive", return_value=False),
            mock.patch.object(enhance.time, "sleep") as sleep,
        ):
            processed = [
                enhance.process_single_item(chain, item, "Chinese")
                for item in items
            ]

        self.assertEqual(len(chain.calls), 1)
        self.assertEqual(built_item_ids, [items[0]["id"]])
        sleep.assert_not_called()
        self.assertEqual(
            [item["AI_status"] for item in processed],
            ["deferred", "deferred", "deferred"],
        )


class ResumableResultTests(EnhanceStateTestCase):
    def result(self, status, error=None):
        item = {
            "id": "2608.12345",
            "AI": complete_ai_fields(),
            "AI_status": status,
            "AI_input_source": "abstract",
        }
        if error is not None:
            item["AI_error"] = error
        return item

    def test_deferred_and_legacy_402_fallback_are_not_resumable(self):
        deferred = self.result(
            "deferred",
            "AI enhancement deferred because the provider balance is exhausted.",
        )
        legacy_402 = self.result(
            "fallback",
            "Error code: 402 - {'error': {'type': 'insufficient_balance_error'}}",
        )

        self.assertFalse(enhance.is_resumable_result(deferred))
        self.assertFalse(enhance.is_resumable_result(legacy_402))

    def test_ok_partial_and_ordinary_422_fallback_are_resumable(self):
        ok = self.result("ok")
        partial = self.result("partial", "structured output was incomplete")
        ordinary_fallback = self.result(
            "fallback",
            "Error code: 422 - {'error': {'type': 'unprocessable_entity_error'}}",
        )

        self.assertTrue(enhance.is_resumable_result(ok))
        self.assertTrue(enhance.is_resumable_result(partial))
        self.assertTrue(enhance.is_resumable_result(ordinary_fallback))


if __name__ == "__main__":
    unittest.main()
