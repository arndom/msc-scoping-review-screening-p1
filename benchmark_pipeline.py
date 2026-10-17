"""Run LLM-assisted title/abstract screening and benchmark it against human labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import openai
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator


SAMPLE_FILENAME = "ground_truth_sample_291.csv"
GROUND_TRUTH_FILENAME = "ground_truth_screened_291.csv"
PROTOCOL_FILENAME = "OSF Protocol Draft - raw.pdf"
OUTPUT_CSV_FILENAME = "llm_screened_291.csv"
OUTPUT_JSON_FILENAME = "benchmark_results.json"
CHECKPOINT_FILENAME = ".benchmark_checkpoint.jsonl"

EXPECTED_RECORDS = 291
MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "none"
SUPPORTED_REASONING_EFFORTS = ("none", "low")
SOURCE_COLUMNS = [
    "S/N",
    "Item Type",
    "Title",
    "Abstract Note",
    "DOI",
    "Url",
    "Language",
]
HUMAN_COLUMNS = ["Decision", "Exclusion Code", "Notes"]
LLM_COLUMNS = ["llm_decision", "llm_exclusion_code", "llm_notes"]
VALID_DECISIONS = {"Include", "Exclude", "Unsure"}
CANDIDATE_DECISIONS = {"Include", "Unsure"}

TEMPERATURE = 0.0
RECALL_TARGET = 0.95
MAX_ATTEMPTS = 4  # One initial request plus three retries.
BACKOFF_SECONDS = (1.0, 2.0, 4.0)
PROMPT_VERSION = "title-abstract-screening-v1.1.1"
RESULTS_SCHEMA_VERSION = "benchmark-results-v1.0.0"


class Decision(str, Enum):
    INCLUDE = "Include"
    EXCLUDE = "Exclude"
    UNSURE = "Unsure"


class ExclusionCode(str, Enum):
    NON_OSS = "EXC-1: Non-OSS Context"
    OUT_OF_SCOPE = "EXC-2: Out of Scope Concept"
    NON_SE = "EXC-3: Non-SE Context"
    GREY = "EXC-4: Grey / Non Peer Reviewed"
    SECONDARY = "EXC-5: Secondary Study"


PHASE_ONE_EXCLUSION_CODES = {code.value for code in ExclusionCode}
ALL_PROTOCOL_EXCLUSION_CODES = PHASE_ONE_EXCLUSION_CODES | {
    "EXC-6: Non-English",
    "EXC-7: Unretrievable",
}


class ScreeningResult(BaseModel):
    """Schema-constrained result for one title-and-abstract record."""

    model_config = ConfigDict(extra="forbid")

    llm_decision: Decision = Field(description="Title/abstract screening decision.")
    llm_exclusion_code: Optional[ExclusionCode] = Field(
        description="First supported phase-one exclusion code, otherwise null."
    )
    llm_notes: str = Field(
        description="One concise line of rationale, or an empty string for Include."
    )

    @model_validator(mode="after")
    def enforce_decision_contract(self) -> "ScreeningResult":
        self.llm_notes = self.llm_notes.strip()
        if "\n" in self.llm_notes or "\r" in self.llm_notes:
            raise ValueError("llm_notes must be a single line")

        if self.llm_decision == Decision.EXCLUDE:
            if self.llm_exclusion_code is None:
                raise ValueError("Exclude requires an exclusion code")
            if not self.llm_notes:
                raise ValueError("Exclude requires a concise rationale")
        elif self.llm_decision == Decision.UNSURE:
            if self.llm_exclusion_code is not None:
                raise ValueError("Unsure must not have an exclusion code")
            if not self.llm_notes:
                raise ValueError("Unsure requires a concise rationale")
        else:
            if self.llm_exclusion_code is not None:
                raise ValueError("Include must not have an exclusion code")
            if self.llm_notes:
                raise ValueError("Include must have an empty rationale")
        return self


SYSTEM_PROMPT = """You are an expert methodological assistant conducting Phase 1 title-and-abstract screening for a PRISMA-ScR compliant scoping review titled "Understanding Abandonment in Open Source Software (OSS)."

Review objective:
Map literature on OSS abandonment to identify operational definitions, underlying causes, study methodologies, and ecosystem impacts.

Population-Concept-Context inclusion criteria:
- Population: Open Source Software (OSS/FLOSS) projects, repositories, or ecosystems.
- Concept: abandonment and synonymous concepts including dormancy, project failure, software decay, inactivity, decline, or unmaintained states.
- Context: software engineering practices, developer activity patterns, software supply-chain impact, project health, and ecosystem sustainability or maintenance dynamics.
- Study type: original empirical research in a peer-reviewed venue.

Concept meaning:
- Scope clarification: the concept includes not only abandonment states, but also explicitly connected causes, predictors, prevention or mitigation mechanisms, resilience factors, and consequences. Treat this clarification as part of the definition used by EXC-2 below.
- Community structure or participation, maintainer support, funding or resources, contributor retention, and long-term support can be relevant when the study analyzes how they help OSS projects continue to be maintained or survive, or how their absence contributes to inactivity or discontinuation. A study can therefore be in scope even when it focuses on avoiding abandonment rather than observing an already-abandoned project.
- Sustainability, project or community health, stability, long-term success, continued development, maintenance capacity, and similar terms can have meanings broader than abandonment. Do not automatically Include a study because one of these terms appears. If one is central to the study's objective, data, analysis, or findings but its relationship to OSS project/repository continuity is unclear, assign Unsure.
- Adjacent factors can be relevant, but do not treat them as lifecycle evidence merely because they could theoretically affect a project. For a study of governance, participation, onboarding, collaboration, code quality, bug handling, developer retention, maintainer workload, funding, or similar factors, assign Unsure only when the title/abstract itself provides a textual bridge to OSS project/repository sustainability, health, continuity, survival, continued maintenance, inactivity, or discontinuation. If the study's actual objective, data, analysis, and findings concern the adjacent topic without such a bridge, assign EXC-2.
- Do not use a sustainability or project-success statement that appears only in general background or motivation as evidence for Include or Unsure. Check whether the study actually investigates that concept or a linked factor through its objectives, data, analysis, or findings.
- Outdated OSS, stale dependencies, delayed updates, or associated risks are not automatically evidence of abandonment. Include them only when the study clearly examines their connection to OSS projects or repositories becoming unsupported, unmaintained, inactive, or discontinued. When this connection is plausible but unclear, assign Unsure; do not require an explicit causal link for Unsure. Assign EXC-2 only when the study clearly concerns ordinary update or dependency practices with no plausible lifecycle connection.
- Distinguish the level of the outcome. Survival, death, churn, or removal of code, components, tables, contributors, or subsystems does not by itself establish project- or repository-level abandonment. If the abstract studies such a lower-level outcome and leaves its relationship to project/repository continuity unclear, assign Unsure.
- Abandonment may be described without using the word “abandonment.” It can include project- or repository-level states and processes such as dormancy, sustained inactivity, maintenance cessation, decay, decline, project death, or reduced likelihood of survival.
- Contributor or maintainer activity, interactions, retention, resources, and effort are not automatically synonymous with abandonment. Treat them as potentially relevant factors, indicators, or consequences when the study’s aims, research questions, analysis, or findings explicitly connect them to OSS project or repository persistence, continued maintenance, inactivity, discontinuation, decline, or survival.
- Assess the study’s purpose, research questions, data, analysis, and outcomes rather than matching isolated terms. Include a record when its research directly examines an abandonment-related state or an explicitly connected factor. If the connection is plausible but unclear from the abstract, assign Unsure. A passing mention in background or motivation alone does not establish relevance.
- Assign EXC-2 only when the title and abstract make it clear that the study’s actual subject is outside these abandonment-related concepts. Do not use the absence of a specific keyword as evidence for exclusion.
- Distinguish project-level inactivity or maintenance cessation from an individual contributor’s inactivity; one does not establish the other without an explicit connection.

Study meaning:
- A primary empirical replication or replication study remains a primary study, even when it reproduces or extends earlier findings using the same dataset. Reserve EXC-5 for literature syntheses such as systematic or scoping reviews, mapping studies, and meta-analyses.
- An empirical analysis of conference talks, practitioner discussions, or other primary data is not a secondary study merely because the data are talks or publications. Reserve EXC-5 for a synthesis of existing studies; if the record does not establish whether it is primary or secondary, assign Unsure rather than inferring EXC-5.
- Do not infer EXC-4 solely from “work-in-progress,” a conference venue, or a missing DOI. Distinguish an individual research paper from a proceedings volume, table of contents, program, or index; do not use a relevant paper listed inside a collection as evidence about the collection-level record.

Apply the following exclusions strictly from top to bottom. Stop at and assign the FIRST criterion for which the supplied title, abstract, or secondary metadata provides explicit evidence; never replace an earlier matching code with a later, more specific code:
- EXC-1: Non-OSS Context — the study population is explicitly outside OSS/FLOSS projects, repositories, or ecosystems. This includes proprietary or closed-source software and clearly non-software populations.
- EXC-2: Out of Scope Concept — assign only when the study's actual objective and evidence are clearly unrelated to OSS project/repository continuity and do not examine an abandonment-related state or a plausible lifecycle factor, indicator, prevention mechanism, or consequence. General maintenance, evolution, quality, or updating clearly unrelated to those concepts is out of scope. An unclear or unproven connection is not, by itself, evidence for EXC-2; use Unsure when the study's focus makes the connection plausible.
- EXC-3: Non-SE Context — only after EXC-1 and EXC-2 do not apply, the study concerns a non-software-engineering domain such as open hardware, networking, open science, or open datasets.
- EXC-4: Grey / Non Peer Reviewed — the item is explicitly non-peer-reviewed or short-form material such as a preprint, keynote, editorial, or slides.
- EXC-5: Secondary Study — the item is secondary or tertiary literature such as a systematic literature review or mapping study rather than a primary study.

High-recall rules:
- If the abstract is ambiguous, incomplete, absent, or lacks explicit evidence for exclusion, assign decision Unsure rather than Exclude.
- Use this three-way threshold: Include when the study clearly examines a project/repository lifecycle outcome or clearly analyzes a factor's relationship to that outcome; Unsure when the study contains a textual lifecycle anchor or bridge but the title/abstract leaves its meaning, relationship, or level of analysis unresolved; EXC-2 when the actual study focus is clearly unrelated and no such anchor or bridge is present. Do not infer a lifecycle relationship solely because a factor might theoretically affect project continuity.
- Do not require explicit causal proof to assign Unsure once a textual lifecycle anchor or bridge is present. If that anchor's meaning or relationship is unresolved, assign Unsure; do not infer a bridge solely because a factor might theoretically matter.
- A broad sustainability, success, health, stability, or long-term-support reference is not sufficient for Include on its own. Check whether the study investigates the concept; if it does but its project-continuity meaning is unclear, assign Unsure.
- Absence of an OSS/open-source keyword alone is not proof that the population is outside OSS; without affirmative evidence, use Unsure rather than EXC-1.
- Explicit evidence that the population is outside OSS must be classified as EXC-1, even when a later criterion such as EXC-3 could also describe the record.
- Do not infer non-peer-reviewed status solely from an unfamiliar venue or missing DOI.
- EXC-6 (Non-English) and EXC-7 (Unretrievable) are disabled in Phase 1 and must never be assigned; they require full-text assessment.
- Do not use outside knowledge. Judge only the supplied record.

Output rules:
- Include: the record appears to satisfy the criteria; exclusion code must be null and notes must be empty.
- Exclude: explicit evidence supports a listed EXC-1 to EXC-5 code; provide the first code and a one-line rationale.
- Unsure: evidence is ambiguous or insufficient; exclusion code must be null and provide a one-line rationale.
"""


class PipelineError(RuntimeError):
    """A user-actionable pipeline error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise PipelineError(f"{label} CSV not found: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    except pd.errors.ParserError as error:
        raise PipelineError(f"Could not parse {label} CSV: {error}") from error


def validate_sample(sample: pd.DataFrame) -> None:
    missing = [column for column in SOURCE_COLUMNS if column not in sample.columns]
    if missing:
        raise PipelineError(
            "Sample CSV is missing required column(s): " + ", ".join(missing)
        )
    if len(sample) != EXPECTED_RECORDS:
        raise PipelineError(
            f"Sample CSV must contain {EXPECTED_RECORDS} records; found {len(sample)}."
        )
    if sample["S/N"].duplicated().any():
        raise PipelineError("Sample CSV contains duplicate S/N values.")
    expected_serials = list(range(1, EXPECTED_RECORDS + 1))
    if sample["S/N"].tolist() != expected_serials:
        raise PipelineError(f"Sample CSV S/N must be the ordered sequence 1-{EXPECTED_RECORDS}.")
    if sample["Title"].astype(str).str.strip().eq("").any():
        raise PipelineError("Sample CSV contains an empty Title.")


def validate_ground_truth(sample: pd.DataFrame, ground_truth: pd.DataFrame) -> None:
    required = [*SOURCE_COLUMNS, *HUMAN_COLUMNS]
    missing = [column for column in required if column not in ground_truth.columns]
    if missing:
        raise PipelineError(
            "Ground-truth CSV is missing required column(s): " + ", ".join(missing)
        )
    if len(ground_truth) != EXPECTED_RECORDS:
        raise PipelineError(
            f"Ground-truth CSV must contain {EXPECTED_RECORDS} records; "
            f"found {len(ground_truth)}."
        )
    if ground_truth["S/N"].duplicated().any():
        raise PipelineError("Ground-truth CSV contains duplicate S/N values.")

    sample_by_id = sample.set_index("S/N").loc[:, SOURCE_COLUMNS[1:]].astype(str)
    try:
        truth_by_id = (
            ground_truth.set_index("S/N")
            .loc[sample_by_id.index, SOURCE_COLUMNS[1:]]
            .astype(str)
        )
    except KeyError as error:
        raise PipelineError("Sample and ground-truth S/N values do not match.") from error
    if set(ground_truth["S/N"]) != set(sample["S/N"]):
        raise PipelineError("Sample and ground-truth S/N values do not match.")
    if not sample_by_id.equals(truth_by_id):
        raise PipelineError("Source fields differ between sample and ground-truth CSVs.")

    decisions = ground_truth["Decision"].astype(str).str.strip()
    codes = ground_truth["Exclusion Code"].astype(str).str.strip()
    invalid_decisions = sorted(set(decisions) - VALID_DECISIONS)
    if invalid_decisions:
        raise PipelineError(
            "Ground-truth CSV contains invalid Decision value(s): "
            + ", ".join(invalid_decisions)
        )
    invalid_codes = sorted(set(codes[codes.ne("")]) - ALL_PROTOCOL_EXCLUSION_CODES)
    if invalid_codes:
        raise PipelineError(
            "Ground-truth CSV contains invalid Exclusion Code value(s): "
            + ", ".join(invalid_codes)
        )
    if ((decisions == "Exclude") & (codes == "")).any():
        raise PipelineError("Every human Exclude decision must have an exclusion code.")
    if ((decisions != "Exclude") & (codes != "")).any():
        raise PipelineError("Human Include/Unsure decisions must not have exclusion codes.")


def build_record_message(record: pd.Series) -> str:
    payload = {column: str(record[column]) for column in SOURCE_COLUMNS}
    return "Screen this record using the supplied hierarchy:\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    )


def is_transient_error(error: Exception) -> bool:
    transient_types = tuple(
        error_type
        for error_type in (
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "InternalServerError", None),
            getattr(openai, "RateLimitError", None),
        )
        if error_type is not None
    )
    if transient_types and isinstance(error, transient_types):
        return True
    status_code = getattr(error, "status_code", None)
    return status_code in {408, 409, 429} or (
        isinstance(status_code, int) and status_code >= 500
    )


def usage_dict(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def refusal_text(response: Any) -> Optional[str]:
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            if getattr(content, "type", None) == "refusal":
                return str(getattr(content, "refusal", "Model refused the request."))
    return None


def classify_record(
    client: Any,
    model: str,
    record: pd.Series,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Classify one record, retrying only transient API failures."""
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise PipelineError(
            "Unsupported reasoning effort: "
            f"{reasoning_effort}. Choose one of: "
            + ", ".join(SUPPORTED_REASONING_EFFORTS)
        )
    started = time.perf_counter()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request: Dict[str, Any] = {
                "model": model,
                "instructions": SYSTEM_PROMPT,
                "input": [{"role": "user", "content": build_record_message(record)}],
                "text_format": ScreeningResult,
                "reasoning": {"effort": reasoning_effort},
                "store": False,
            }
            if reasoning_effort == "none":
                request["temperature"] = TEMPERATURE
            response = client.responses.parse(**request)
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                refusal = refusal_text(response)
                if refusal:
                    raise PipelineError(f"Model refusal: {refusal}")
                raise PipelineError("API response did not contain a parsed result.")
            if not isinstance(parsed, ScreeningResult):
                parsed = ScreeningResult.model_validate(parsed)
            return {
                "result": parsed.model_dump(mode="json"),
                "response_id": str(getattr(response, "id", "")),
                "returned_model": str(getattr(response, "model", model)),
                "usage": usage_dict(response),
                "latency_seconds": round(time.perf_counter() - started, 6),
                "attempts": attempt,
            }
        except Exception as error:
            if not is_transient_error(error) or attempt == MAX_ATTEMPTS:
                raise
            sleep(BACKOFF_SECONDS[attempt - 1])
    raise AssertionError("Retry loop exited unexpectedly.")


def run_signature(
    sample_hash: str, protocol_hash: str, model: str, reasoning_effort: str
) -> str:
    signature_data = {
        "sample_sha256": sample_hash,
        "protocol_sha256": protocol_hash,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "temperature": TEMPERATURE if reasoning_effort == "none" else None,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "schema": ScreeningResult.model_json_schema(),
    }
    serialized = json.dumps(
        signature_data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256_text(serialized)


def initialize_checkpoint(
    path: Path,
    signature: str,
    model: str,
    reasoning_effort: str,
    restart: bool,
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    if path.exists() and not restart:
        return load_checkpoint(path, signature)

    header = {
        "kind": "metadata",
        "signature": signature,
        "created_at_utc": utc_now(),
        "model_requested": model,
        "reasoning_effort": reasoning_effort,
        "temperature": TEMPERATURE if reasoning_effort == "none" else None,
        "prompt_version": PROMPT_VERSION,
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return header, {}


def load_checkpoint(
    path: Path, expected_signature: str
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    try:
        entries = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"Checkpoint is unreadable: {error}") from error
    if not entries or entries[0].get("kind") != "metadata":
        raise PipelineError("Checkpoint is missing its metadata header.")
    header = entries[0]
    if header.get("signature") != expected_signature:
        raise PipelineError(
            "Checkpoint does not match the current input, prompt, protocol, or model. "
            "Run with --restart to start a new benchmark."
        )

    records: Dict[int, Dict[str, Any]] = {}
    for entry in entries[1:]:
        if entry.get("kind") != "record" or "S/N" not in entry:
            raise PipelineError("Checkpoint contains an invalid entry.")
        serial = int(entry["S/N"])
        if serial in records:
            raise PipelineError(f"Checkpoint contains duplicate S/N {serial}.")
        ScreeningResult.model_validate(entry.get("result"))
        records[serial] = entry
    return header, records


def append_checkpoint(path: Path, entry: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def metric(numerator: int, denominator: int) -> Dict[str, Any]:
    return {
        "value": None if denominator == 0 else round(numerator / denominator, 6),
        "numerator": int(numerator),
        "denominator": int(denominator),
    }


def compute_metrics(
    ground_truth: pd.DataFrame, predictions: pd.DataFrame
) -> Dict[str, Any]:
    merged = ground_truth[["S/N", "Decision", "Exclusion Code"]].merge(
        predictions[["S/N", *LLM_COLUMNS]], on="S/N", how="inner", validate="one_to_one"
    )
    if len(merged) != len(ground_truth):
        raise PipelineError("Predictions do not cover every ground-truth record.")

    human_candidate = merged["Decision"].isin(CANDIDATE_DECISIONS)
    llm_candidate = merged["llm_decision"].isin(CANDIDATE_DECISIONS)
    tp = int((human_candidate & llm_candidate).sum())
    fn = int((human_candidate & ~llm_candidate).sum())
    fp = int((~human_candidate & llm_candidate).sum())
    tn = int((~human_candidate & ~llm_candidate).sum())

    recall = metric(tp, tp + fn)
    three_class_correct = int((merged["Decision"] == merged["llm_decision"]).sum())
    human_excluded = merged["Decision"] == "Exclude"
    code_correct = int(
        (
            human_excluded
            & (merged["llm_decision"] == "Exclude")
            & (merged["Exclusion Code"] == merged["llm_exclusion_code"])
        ).sum()
    )

    return {
        "candidate_positive_definition": ["Include", "Unsure"],
        "candidate_confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "recall_sensitivity": recall,
        "recall_target": RECALL_TARGET,
        "recall_target_met": (
            recall["value"] is not None and recall["value"] >= RECALL_TARGET
        ),
        "false_negative_rate": metric(fn, tp + fn),
        "precision": metric(tp, tp + fp),
        "specificity": metric(tn, tn + fp),
        "binary_accuracy": metric(tp + tn, len(merged)),
        "three_class_decision_accuracy": metric(three_class_correct, len(merged)),
        "exclusion_code_accuracy_on_human_excludes": metric(
            code_correct, int(human_excluded.sum())
        ),
        "human_class_counts": {
            key: int(value)
            for key, value in merged["Decision"].value_counts().sort_index().items()
        },
        "llm_class_counts": {
            key: int(value)
            for key, value in merged["llm_decision"].value_counts().sort_index().items()
        },
        "false_negative_record_ids": [
            int(value)
            for value in merged.loc[human_candidate & ~llm_candidate, "S/N"].tolist()
        ],
        "false_positive_record_ids": [
            int(value)
            for value in merged.loc[~human_candidate & llm_candidate, "S/N"].tolist()
        ],
    }


def predictions_from_checkpoint(
    sample: pd.DataFrame, checkpoint_records: Dict[int, Dict[str, Any]]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for serial in sample["S/N"]:
        entry = checkpoint_records.get(int(serial))
        if entry is None:
            raise PipelineError(f"Checkpoint is missing prediction for S/N {serial}.")
        rows.append({"S/N": int(serial), **entry["result"]})
    return pd.DataFrame(rows, columns=["S/N", *LLM_COLUMNS])


def aggregate_usage(checkpoint_records: Dict[int, Dict[str, Any]]) -> Dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for entry in checkpoint_records.values():
        for key in totals:
            totals[key] += int(entry.get("usage", {}).get(key, 0))
    return totals


def atomic_write_csv(data: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    data.to_csv(
        temporary, index=False, encoding="utf-8", lineterminator="\n"
    )
    os.replace(temporary, path)


def atomic_write_json(data: Dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def build_report(
    base_directory: Path,
    model: str,
    reasoning_effort: str,
    signature: str,
    header: Dict[str, Any],
    checkpoint_records: Dict[int, Dict[str, Any]],
    metrics: Dict[str, Any],
    resumed_records: int,
    invocation_seconds: float,
) -> Dict[str, Any]:
    sample_path = base_directory / SAMPLE_FILENAME
    ground_truth_path = base_directory / GROUND_TRUTH_FILENAME
    protocol_path = base_directory / PROTOCOL_FILENAME
    returned_models = sorted(
        {
            str(entry.get("returned_model", ""))
            for entry in checkpoint_records.values()
            if entry.get("returned_model")
        }
    )
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "run": {
            "signature": signature,
            "started_at_utc": header["created_at_utc"],
            "completed_at_utc": utc_now(),
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": sha256_text(SYSTEM_PROMPT),
            "prompt": SYSTEM_PROMPT,
            "protocol": {
                "filename": PROTOCOL_FILENAME,
                "sha256": sha256_file(protocol_path),
            },
            "inputs": {
                SAMPLE_FILENAME: sha256_file(sample_path),
                GROUND_TRUTH_FILENAME: sha256_file(ground_truth_path),
            },
            "model_requested": model,
            "models_returned": returned_models,
            "reasoning_effort": reasoning_effort,
            "temperature": TEMPERATURE if reasoning_effort == "none" else None,
            "store": False,
            "records": len(checkpoint_records),
            "resumed_records": resumed_records,
            "new_records": len(checkpoint_records) - resumed_records,
            "total_api_latency_seconds": round(
                sum(
                    float(entry.get("latency_seconds", 0.0))
                    for entry in checkpoint_records.values()
                ),
                6,
            ),
            "current_invocation_seconds": round(invocation_seconds, 6),
            "token_usage": aggregate_usage(checkpoint_records),
            "dependencies": {
                "openai": version("openai"),
                "pandas": version("pandas"),
                "pydantic": version("pydantic"),
            },
        },
        "metrics": metrics,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify and benchmark the 291-record screening sample."
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard a prior checkpoint and classify all records again.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=SUPPORTED_REASONING_EFFORTS,
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "GPT-6 Luna reasoning effort (default: none). Temperature 0.0 is sent "
            "with none and omitted with low."
        ),
    )
    return parser.parse_args(argv)


def execute(
    base_directory: Path,
    model: str = MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    restart: bool = False,
) -> None:
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise PipelineError(
            "Unsupported reasoning effort: "
            f"{reasoning_effort}. Choose one of: "
            + ", ".join(SUPPORTED_REASONING_EFFORTS)
        )
    invocation_started = time.perf_counter()
    sample_path = base_directory / SAMPLE_FILENAME
    ground_truth_path = base_directory / GROUND_TRUTH_FILENAME
    protocol_path = base_directory / PROTOCOL_FILENAME
    checkpoint_path = base_directory / CHECKPOINT_FILENAME

    if not protocol_path.is_file():
        raise PipelineError(f"Protocol PDF not found: {protocol_path}")
    sample = read_csv(sample_path, "Sample")
    ground_truth = read_csv(ground_truth_path, "Ground-truth")
    validate_sample(sample)
    validate_ground_truth(sample, ground_truth)

    signature = run_signature(
        sha256_file(sample_path),
        sha256_file(protocol_path),
        model,
        reasoning_effort,
    )
    header, checkpoint_records = initialize_checkpoint(
        checkpoint_path, signature, model, reasoning_effort, restart
    )
    resumed_records = len(checkpoint_records)
    client = OpenAI()

    for _, record in sample.iterrows():
        serial = int(record["S/N"])
        if serial in checkpoint_records:
            continue
        print(f"Screening S/N {serial}/{EXPECTED_RECORDS}...", flush=True)
        api_result = classify_record(
            client, model, record, reasoning_effort=reasoning_effort
        )
        entry = {"kind": "record", "S/N": serial, **api_result}
        append_checkpoint(checkpoint_path, entry)
        checkpoint_records[serial] = entry

    predictions = predictions_from_checkpoint(sample, checkpoint_records)
    screened = sample.loc[:, SOURCE_COLUMNS].merge(
        predictions, on="S/N", how="left", validate="one_to_one"
    )
    if screened["llm_decision"].isna().any().any():
        raise PipelineError("Generated screening CSV contains missing LLM results.")
    metrics = compute_metrics(ground_truth, predictions)
    report = build_report(
        base_directory,
        model,
        reasoning_effort,
        signature,
        header,
        checkpoint_records,
        metrics,
        resumed_records,
        time.perf_counter() - invocation_started,
    )
    atomic_write_csv(screened, base_directory / OUTPUT_CSV_FILENAME)
    atomic_write_json(report, base_directory / OUTPUT_JSON_FILENAME)

    recall = metrics["recall_sensitivity"]["value"]
    recall_text = "undefined" if recall is None else f"{recall:.2%}"
    print(
        f"Completed {EXPECTED_RECORDS} records. Candidate recall: {recall_text}. "
        f"Wrote {OUTPUT_CSV_FILENAME} and {OUTPUT_JSON_FILENAME}."
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    base_directory = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=base_directory / ".env", override=False)
    missing_environment = [
        name for name in ("OPENAI_API_KEY",) if not os.environ.get(name)
    ]
    if missing_environment:
        print(
            "Error: missing required environment variable(s): "
            + ", ".join(missing_environment),
            file=sys.stderr,
        )
        return 2

    try:
        execute(
            base_directory,
            model=MODEL,
            reasoning_effort=args.reasoning_effort,
            restart=args.restart,
        )
    except (PipelineError, OSError, ValueError, openai.OpenAIError) as error:
        print(f"Error: {error}", file=sys.stderr)
        print(
            f"Any successful records remain in {CHECKPOINT_FILENAME}; rerun to resume.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
