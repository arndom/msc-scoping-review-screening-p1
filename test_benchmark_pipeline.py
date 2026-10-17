"""Offline tests for the LLM screening and benchmarking pipeline."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from pydantic import ValidationError

import benchmark_pipeline as pipeline


def make_sample() -> pd.DataFrame:
    count = pipeline.EXPECTED_RECORDS
    return pd.DataFrame(
        {
            "S/N": range(1, count + 1),
            "Item Type": ["journalArticle"] * count,
            "Title": [f"Title {number}" for number in range(1, count + 1)],
            "Abstract Note": [f"Abstract {number}" for number in range(1, count + 1)],
            "DOI": [""] * count,
            "Url": [""] * count,
            "Language": [""] * count,
        }
    )


def make_ground_truth(sample: pd.DataFrame) -> pd.DataFrame:
    ground_truth = sample.copy()
    ground_truth["Decision"] = "Exclude"
    ground_truth["Exclusion Code"] = pipeline.ExclusionCode.OUT_OF_SCOPE.value
    ground_truth["Notes"] = ""
    return ground_truth


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.responses = FakeResponses(outcomes)


def fake_response(result: pipeline.ScreeningResult):
    return SimpleNamespace(
        output_parsed=result,
        id="resp_test",
        model="test-model-snapshot",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        output=[],
    )


class ScreeningResultTests(unittest.TestCase):
    def test_valid_results(self):
        include = pipeline.ScreeningResult(
            llm_decision="Include", llm_exclusion_code=None, llm_notes=""
        )
        unsure = pipeline.ScreeningResult(
            llm_decision="Unsure",
            llm_exclusion_code=None,
            llm_notes="Evidence is incomplete.",
        )
        exclude = pipeline.ScreeningResult(
            llm_decision="Exclude",
            llm_exclusion_code="EXC-2: Out of Scope Concept",
            llm_notes="The study addresses testing but not abandonment.",
        )
        self.assertEqual(include.llm_decision.value, "Include")
        self.assertEqual(unsure.llm_notes, "Evidence is incomplete.")
        self.assertEqual(exclude.llm_exclusion_code.value, pipeline.ExclusionCode.OUT_OF_SCOPE.value)

    def test_conditional_contract_rejects_invalid_combinations(self):
        invalid = [
            {
                "llm_decision": "Exclude",
                "llm_exclusion_code": None,
                "llm_notes": "Reason",
            },
            {
                "llm_decision": "Unsure",
                "llm_exclusion_code": "EXC-1: Non-OSS Context",
                "llm_notes": "Reason",
            },
            {
                "llm_decision": "Include",
                "llm_exclusion_code": None,
                "llm_notes": "Unexpected note",
            },
            {
                "llm_decision": "Unsure",
                "llm_exclusion_code": None,
                "llm_notes": "Two\nlines",
            },
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                pipeline.ScreeningResult(**values)


class PromptHierarchyTests(unittest.TestCase):
    def test_exclusion_hierarchy_and_non_oss_precedence_are_explicit(self):
        prompt = pipeline.SYSTEM_PROMPT
        exc_1 = prompt.index("- EXC-1:")
        exc_2 = prompt.index("- EXC-2:")
        exc_3 = prompt.index("- EXC-3:")
        self.assertLess(exc_1, exc_2)
        self.assertLess(exc_2, exc_3)
        self.assertIn("Stop at and assign the FIRST criterion", prompt)
        self.assertIn(
            "population is outside OSS must be classified as EXC-1", prompt
        )


class ValidationTests(unittest.TestCase):
    def test_valid_data(self):
        sample = make_sample()
        truth = make_ground_truth(sample)
        pipeline.validate_sample(sample)
        pipeline.validate_ground_truth(sample, truth)

    def test_sample_validation_failures(self):
        sample = make_sample()
        with self.subTest("missing column"), self.assertRaises(pipeline.PipelineError):
            pipeline.validate_sample(sample.drop(columns=["Title"]))
        duplicate = sample.copy()
        duplicate.loc[1, "S/N"] = 1
        with self.subTest("duplicate serial"), self.assertRaises(pipeline.PipelineError):
            pipeline.validate_sample(duplicate)
        wrong_count = sample.iloc[:-1].copy()
        with self.subTest("wrong count"), self.assertRaises(pipeline.PipelineError):
            pipeline.validate_sample(wrong_count)

    def test_ground_truth_validation_failures(self):
        sample = make_sample()
        truth = make_ground_truth(sample)

        mismatched = truth.copy()
        mismatched.loc[0, "Title"] = "Changed"
        with self.subTest("source mismatch"), self.assertRaises(pipeline.PipelineError):
            pipeline.validate_ground_truth(sample, mismatched)

        invalid_decision = truth.copy()
        invalid_decision.loc[0, "Decision"] = "Maybe"
        with self.subTest("invalid decision"), self.assertRaises(pipeline.PipelineError):
            pipeline.validate_ground_truth(sample, invalid_decision)

        missing_code = truth.copy()
        missing_code.loc[0, "Exclusion Code"] = ""
        with self.subTest("missing exclusion code"), self.assertRaises(pipeline.PipelineError):
            pipeline.validate_ground_truth(sample, missing_code)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.record = make_sample().iloc[0]

    def test_api_payload_and_usage(self):
        result = pipeline.ScreeningResult(
            llm_decision="Unsure",
            llm_exclusion_code=None,
            llm_notes="The abstract does not establish the study context.",
        )
        client = FakeClient([fake_response(result)])
        classified = pipeline.classify_record(client, "test-model", self.record)

        self.assertEqual(classified["result"]["llm_decision"], "Unsure")
        self.assertEqual(classified["usage"]["total_tokens"], 15)
        kwargs = client.responses.calls[0]
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["reasoning"], {"effort": "none"})
        self.assertFalse(kwargs["store"])
        self.assertIs(kwargs["text_format"], pipeline.ScreeningResult)
        record_payload = kwargs["input"][0]["content"]
        for column in pipeline.SOURCE_COLUMNS:
            self.assertIn(f'"{column}"', record_payload)
        self.assertNotIn('"Decision"', record_payload)
        self.assertNotIn('"Exclusion Code"', record_payload)
        self.assertNotIn('"Notes"', record_payload)

    def test_low_reasoning_omits_temperature(self):
        result = pipeline.ScreeningResult(
            llm_decision="Include", llm_exclusion_code=None, llm_notes=""
        )
        client = FakeClient([fake_response(result)])

        pipeline.classify_record(
            client, "test-model", self.record, reasoning_effort="low"
        )

        kwargs = client.responses.calls[0]
        self.assertEqual(kwargs["reasoning"], {"effort": "low"})
        self.assertNotIn("temperature", kwargs)

    def test_unsupported_reasoning_effort_is_rejected(self):
        client = FakeClient([])
        with self.assertRaisesRegex(pipeline.PipelineError, "Unsupported"):
            pipeline.classify_record(
                client, "test-model", self.record, reasoning_effort="medium"
            )
        self.assertEqual(client.responses.calls, [])

    def test_transient_failure_is_retried(self):
        result = pipeline.ScreeningResult(
            llm_decision="Include", llm_exclusion_code=None, llm_notes=""
        )
        transient = RuntimeError("temporary")
        client = FakeClient([transient, fake_response(result)])
        delays = []
        with patch.object(
            pipeline, "is_transient_error", side_effect=lambda error: error is transient
        ):
            classified = pipeline.classify_record(
                client, "test-model", self.record, sleep=delays.append
            )
        self.assertEqual(classified["attempts"], 2)
        self.assertEqual(delays, [1.0])

    def test_refusal_fails_without_retry(self):
        refusal = SimpleNamespace(
            output_parsed=None,
            output=[
                SimpleNamespace(
                    content=[SimpleNamespace(type="refusal", refusal="Cannot comply")]
                )
            ],
        )
        client = FakeClient([refusal])
        with self.assertRaisesRegex(pipeline.PipelineError, "Model refusal"):
            pipeline.classify_record(client, "test-model", self.record)
        self.assertEqual(len(client.responses.calls), 1)


class CheckpointTests(unittest.TestCase):
    def test_reasoning_effort_changes_signature(self):
        none_signature = pipeline.run_signature("sample", "protocol", "model", "none")
        low_signature = pipeline.run_signature("sample", "protocol", "model", "low")
        self.assertNotEqual(none_signature, low_signature)

    def test_checkpoint_resume_and_signature_mismatch(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            header, records = pipeline.initialize_checkpoint(
                path, "signature-one", "model", "none", restart=False
            )
            self.assertEqual(records, {})
            result = pipeline.ScreeningResult(
                llm_decision="Include", llm_exclusion_code=None, llm_notes=""
            )
            entry = {
                "kind": "record",
                "S/N": 1,
                "result": result.model_dump(mode="json"),
                "usage": {},
            }
            pipeline.append_checkpoint(path, entry)
            loaded_header, loaded_records = pipeline.load_checkpoint(
                path, "signature-one"
            )
            self.assertEqual(loaded_header, header)
            self.assertEqual(loaded_records[1]["result"]["llm_decision"], "Include")
            with self.assertRaisesRegex(pipeline.PipelineError, "--restart"):
                pipeline.load_checkpoint(path, "signature-two")


class MetricTests(unittest.TestCase):
    def test_metrics_and_record_ids(self):
        truth = pd.DataFrame(
            {
                "S/N": [1, 2, 3, 4, 5],
                "Decision": ["Include", "Unsure", "Exclude", "Exclude", "Exclude"],
                "Exclusion Code": [
                    "",
                    "",
                    pipeline.ExclusionCode.NON_OSS.value,
                    pipeline.ExclusionCode.OUT_OF_SCOPE.value,
                    pipeline.ExclusionCode.GREY.value,
                ],
            }
        )
        predictions = pd.DataFrame(
            {
                "S/N": [1, 2, 3, 4, 5],
                "llm_decision": ["Include", "Exclude", "Unsure", "Exclude", "Exclude"],
                "llm_exclusion_code": [
                    None,
                    pipeline.ExclusionCode.OUT_OF_SCOPE.value,
                    None,
                    pipeline.ExclusionCode.OUT_OF_SCOPE.value,
                    pipeline.ExclusionCode.NON_OSS.value,
                ],
                "llm_notes": ["", "reason", "unclear", "reason", "reason"],
            }
        )
        metrics = pipeline.compute_metrics(truth, predictions)
        self.assertEqual(
            metrics["candidate_confusion_matrix"],
            {"tp": 1, "fn": 1, "fp": 1, "tn": 2},
        )
        self.assertEqual(metrics["recall_sensitivity"]["value"], 0.5)
        self.assertEqual(metrics["false_negative_rate"]["value"], 0.5)
        self.assertEqual(metrics["precision"]["value"], 0.5)
        self.assertEqual(metrics["specificity"]["value"], round(2 / 3, 6))
        self.assertEqual(metrics["false_negative_record_ids"], [2])
        self.assertEqual(metrics["false_positive_record_ids"], [3])
        self.assertEqual(
            metrics["exclusion_code_accuracy_on_human_excludes"]["value"],
            round(1 / 3, 6),
        )

    def test_metric_zero_denominator_and_threshold(self):
        undefined = pipeline.metric(0, 0)
        self.assertIsNone(undefined["value"])
        target = pipeline.metric(95, 100)
        self.assertGreaterEqual(target["value"], pipeline.RECALL_TARGET)


class SerializationTests(unittest.TestCase):
    def test_json_serialization_is_stable(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            payload = {"z": 1, "a": {"é": "value"}}
            pipeline.atomic_write_json(payload, path)
            first = path.read_bytes()
            pipeline.atomic_write_json(payload, path)
            second = path.read_bytes()
            self.assertEqual(first, second)
            self.assertEqual(json.loads(first), payload)

    def test_main_requires_environment(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            pipeline, "load_dotenv"
        ):
            self.assertEqual(pipeline.main([]), 2)

    def test_main_loads_dotenv_without_overriding_process_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), patch.object(
            pipeline, "load_dotenv"
        ) as mocked_load, patch.object(pipeline, "execute") as mocked_execute:
            self.assertEqual(pipeline.main([]), 0)

        expected_directory = Path(pipeline.__file__).resolve().parent
        mocked_load.assert_called_once_with(
            dotenv_path=expected_directory / ".env", override=False
        )
        mocked_execute.assert_called_once_with(
            expected_directory,
            model=pipeline.MODEL,
            reasoning_effort=pipeline.DEFAULT_REASONING_EFFORT,
            restart=False,
        )


if __name__ == "__main__":
    unittest.main()
