import copy
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "ai"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

import enhance  # noqa: E402


def ai_fields(details):
    return {
        "tldr": details,
        "motivation": details,
        "method": details,
        "result": details,
        "conclusion": details,
    }


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class CheckpointPreservationTests(unittest.TestCase):
    def test_interrupt_preserves_unprocessed_deferred_checkpoint_rows(self):
        source_rows = [
            {"id": "paper-ok", "title": "Already complete", "summary": "one"},
            {"id": "paper-retry-1", "title": "First retry", "summary": "two"},
            {"id": "paper-retry-2", "title": "Second retry", "summary": "three"},
        ]
        existing_ok = {
            **source_rows[0],
            "AI": ai_fields("existing success"),
            "AI_status": "ok",
            "AI_input_source": "abstract",
        }
        first_deferred = {
            **source_rows[1],
            "AI": ai_fields("temporarily unavailable"),
            "AI_status": "deferred",
            "AI_error": "Error code: 402 - insufficient_balance_error",
            "AI_input_source": "abstract",
        }
        second_deferred = {
            **source_rows[2],
            "AI": ai_fields("temporarily unavailable"),
            "AI_status": "deferred",
            "AI_error": "AI enhancement deferred because the circuit is open.",
            "AI_input_source": "abstract",
        }

        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "enhanced.jsonl"
            write_jsonl(
                checkpoint_path,
                [existing_ok, first_deferred, second_deferred],
            )
            raw_ids = {row["id"] for row in source_rows}

            checkpoint_by_id = enhance.load_checkpoint_results(
                str(checkpoint_path),
                raw_ids,
            )
            processed_by_id = enhance.load_existing_results(
                str(checkpoint_path),
                raw_ids,
            )

            self.assertEqual(set(checkpoint_by_id), raw_ids)
            self.assertEqual(set(processed_by_id), {"paper-ok"})

            processed_calls = []

            def process_item(_chain, item, _language):
                processed_calls.append(item["id"])
                if item["id"] == "paper-retry-1":
                    result = copy.deepcopy(item)
                    result.update(
                        {
                            "AI": ai_fields("new success"),
                            "AI_status": "ok",
                            "AI_input_source": "abstract",
                        }
                    )
                    return result
                raise KeyboardInterrupt("simulated workflow interruption")

            fake_chain = object()
            with (
                mock.patch.object(enhance, "AI_MAX_SECONDS", 3600.0),
                mock.patch.object(enhance, "AI_INPUT_SOURCE", "abstract"),
                mock.patch.object(
                    enhance,
                    "build_chain",
                    return_value=fake_chain,
                ) as build_chain,
                mock.patch.object(
                    enhance,
                    "process_single_item",
                    side_effect=process_item,
                ),
                mock.patch.object(
                    enhance,
                    "progress_iter",
                    side_effect=lambda iterable, **_kwargs: iterable,
                ),
                mock.patch.object(
                    enhance,
                    "is_ai_time_budget_exceeded",
                    return_value=False,
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    enhance.process_all_items(
                        source_rows,
                        "test-model",
                        "Chinese",
                        1,
                        str(checkpoint_path),
                        processed_by_id,
                        checkpoint_by_id,
                    )

            build_chain.assert_called_once_with("test-model")
            self.assertEqual(processed_calls, ["paper-retry-1", "paper-retry-2"])

            written_rows = [
                json.loads(line)
                for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            written_by_id = {row["id"]: row for row in written_rows}

            self.assertEqual(len(written_rows), 3)
            self.assertEqual(set(written_by_id), raw_ids)
            self.assertEqual(written_by_id["paper-ok"], existing_ok)
            self.assertEqual(written_by_id["paper-retry-1"]["AI_status"], "ok")
            self.assertEqual(
                written_by_id["paper-retry-1"]["AI"],
                ai_fields("new success"),
            )
            self.assertEqual(written_by_id["paper-retry-2"], second_deferred)


if __name__ == "__main__":
    unittest.main()
