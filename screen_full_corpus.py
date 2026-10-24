"""Apply the benchmark's exact screening prompt to the full raw corpus."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

import benchmark_pipeline as benchmark


INPUT_FILENAME = "raw_records_2913.csv"
OUTPUT_CSV_FILENAME = "llm_screened_2913.csv"
OUTPUT_JSON_FILENAME = "llm_screening_report_2913.json"
CHECKPOINT_FILENAME = ".full_corpus_checkpoint.jsonl"
EXPECTED_RECORDS = 2913
REPORT_SCHEMA_VERSION = "full-screening-report-v1.0.0"
OUTPUT_COLUMNS = [
    "S/N",
    "Item Type",
    "Publication Year",
    "Author",
    "Title",
    "DOI",
    "Url",
    "Abstract Note",
    "Decision",
    "Exclusion Code",
    "Notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise benchmark.PipelineError(f"Raw input CSV not found: {path}")
    try:
        data = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    except pd.errors.ParserError as error:
        raise benchmark.PipelineError(f"Could not parse raw input CSV: {error}") from error
    if len(data) != EXPECTED_RECORDS:
        raise benchmark.PipelineError(
            f"Raw input must contain {EXPECTED_RECORDS} records; found {len(data)}."
        )
    missing = [column for column in benchmark.SOURCE_COLUMNS[1:] if column not in data]
    if missing:
        raise benchmark.PipelineError(
            "Raw input is missing required screening column(s): " + ", ".join(missing)
        )
    if data["Title"].astype(str).str.strip().eq("").any():
        raise benchmark.PipelineError("Raw input contains an empty Title.")
    # Raw exports do not have the benchmark's sequential sample identifier.
    data.insert(0, "S/N", range(1, len(data) + 1))
    return data


def signature_for(input_hash: str, protocol_hash: str, model: str, effort: str) -> str:
    config = {
        "input_sha256": input_hash,
        "protocol_sha256": protocol_hash,
        "model": model,
        "reasoning_effort": effort,
        "temperature": benchmark.TEMPERATURE if effort == "none" else None,
        "prompt_version": benchmark.PROMPT_VERSION,
        "prompt_sha256": benchmark.sha256_text(benchmark.SYSTEM_PROMPT),
        "schema": benchmark.ScreeningResult.model_json_schema(),
    }
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_or_create_checkpoint(
    path: Path, signature: str, model: str, effort: str, restart: bool
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    if path.exists() and not restart:
        return benchmark.load_checkpoint(path, signature)
    header = {
        "kind": "metadata",
        "signature": signature,
        "created_at_utc": utc_now(),
        "model_requested": model,
        "reasoning_effort": effort,
        "temperature": benchmark.TEMPERATURE if effort == "none" else None,
        "prompt_version": benchmark.PROMPT_VERSION,
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(header, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return header, {}


def count_results(results: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    decisions = results["Decision"].value_counts().to_dict()
    code_column = results["Exclusion Code"].fillna("")
    codes = code_column.value_counts().to_dict()
    decision_counts = {label: int(decisions.get(label, 0)) for label in ("Include", "Unsure", "Exclude")}
    code_counts = {
        code.value: int(codes.get(code.value, 0))
        for code in benchmark.ExclusionCode
    }
    code_counts["No exclusion code"] = int(code_column.eq("").sum())
    return decision_counts, code_counts


def decision_percentages(decisions: Dict[str, int], total: int) -> Dict[str, Any]:
    candidates = decisions["Include"] + decisions["Unsure"]
    categories = {
        "Exclude": decisions["Exclude"],
        "Include": decisions["Include"],
        "Unsure": decisions["Unsure"],
        "Include_or_Unsure": candidates,
    }
    return {
        label: {
            "percentage": None if total == 0 else round(count * 100 / total, 2),
            "count": count,
            "denominator": total,
        }
        for label, count in categories.items()
    }


def build_output(raw: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    source_columns = OUTPUT_COLUMNS[:8]
    # The raw export also has a `Notes` field. Exclude it deliberately so the
    # output's `Notes` column is unambiguously the LLM screening rationale.
    source_data = raw.loc[:, source_columns]
    combined = source_data.merge(predictions, on="S/N", how="left", validate="one_to_one")
    if len(combined) != EXPECTED_RECORDS:
        raise benchmark.PipelineError("Generated output does not contain all raw records.")
    if combined[["llm_decision", "llm_notes"]].isna().any().any():
        raise benchmark.PipelineError("Generated output contains missing LLM decisions or notes.")
    combined = combined.rename(
        columns={
            "llm_decision": "Decision",
            "llm_exclusion_code": "Exclusion Code",
            "llm_notes": "Notes",
        }
    )
    return combined.loc[:, OUTPUT_COLUMNS]


def write_json(path: Path, report: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def execute(
    base_directory: Path,
    model: str = benchmark.MODEL,
    reasoning_effort: str = benchmark.DEFAULT_REASONING_EFFORT,
    restart: bool = False,
    client: Any = None,
) -> Dict[str, Any]:
    if reasoning_effort not in benchmark.SUPPORTED_REASONING_EFFORTS:
        raise benchmark.PipelineError("Unsupported reasoning effort: " + reasoning_effort)
    input_path = base_directory / INPUT_FILENAME
    protocol_path = base_directory / benchmark.PROTOCOL_FILENAME
    if not protocol_path.is_file():
        raise benchmark.PipelineError(f"Protocol PDF not found: {protocol_path}")
    raw = read_input(input_path)
    # The model receives the same seven fields and the exact same prompt/schema as
    # benchmark_pipeline.py; all other raw bibliographic columns are retained only
    # in the output CSV and are not sent to the API.
    screened_input = raw.loc[:, benchmark.SOURCE_COLUMNS]
    signature = signature_for(
        sha256_file(input_path), sha256_file(protocol_path), model, reasoning_effort
    )
    checkpoint_path = base_directory / CHECKPOINT_FILENAME
    header, completed = load_or_create_checkpoint(
        checkpoint_path, signature, model, reasoning_effort, restart
    )
    resumed = len(completed)
    api_client = client if client is not None else OpenAI()
    started = time.perf_counter()

    for _, record in screened_input.iterrows():
        serial = int(record["S/N"])
        if serial in completed:
            continue
        print(f"Screening S/N {serial}/{EXPECTED_RECORDS}...", flush=True)
        result = benchmark.classify_record(
            api_client, model, record, reasoning_effort=reasoning_effort
        )
        entry = {"kind": "record", "S/N": serial, **result}
        benchmark.append_checkpoint(checkpoint_path, entry)
        completed[serial] = entry

    predictions = benchmark.predictions_from_checkpoint(screened_input, completed)
    output = build_output(raw, predictions)
    decisions, codes = count_results(output)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run": {
            "signature": signature,
            "started_at_utc": header["created_at_utc"],
            "completed_at_utc": utc_now(),
            "input_filename": INPUT_FILENAME,
            "input_sha256": sha256_file(input_path),
            "protocol_filename": benchmark.PROTOCOL_FILENAME,
            "protocol_sha256": sha256_file(protocol_path),
            "prompt_version": benchmark.PROMPT_VERSION,
            "prompt_sha256": benchmark.sha256_text(benchmark.SYSTEM_PROMPT),
            "prompt": benchmark.SYSTEM_PROMPT,
            "model_requested": model,
            "models_returned": sorted({str(entry.get("returned_model", model)) for entry in completed.values()}),
            "reasoning_effort": reasoning_effort,
            "temperature": benchmark.TEMPERATURE if reasoning_effort == "none" else None,
            "store": False,
            "records": len(output),
            "resumed_records": resumed,
            "new_records": len(completed) - resumed,
            "token_usage": benchmark.aggregate_usage(completed),
            "total_api_latency_seconds": round(sum(float(e.get("latency_seconds", 0)) for e in completed.values()), 6),
            "current_invocation_seconds": round(time.perf_counter() - started, 6),
            "dependencies": {
                "openai": version("openai"),
                "pandas": version("pandas"),
                "pydantic": version("pydantic"),
            },
        },
        "decision_counts": decisions,
        "decision_percentages": decision_percentages(decisions, len(output)),
        "exclusion_code_counts": codes,
        "total_excluded": decisions["Exclude"],
        "total_candidates_include_or_unsure": decisions["Include"] + decisions["Unsure"],
    }
    benchmark.atomic_write_csv(output, base_directory / OUTPUT_CSV_FILENAME)
    write_json(base_directory / OUTPUT_JSON_FILENAME, report)
    print(
        f"Completed {len(output)} records. Decisions: {decisions}. "
        f"Wrote {OUTPUT_CSV_FILENAME} and {OUTPUT_JSON_FILENAME}."
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Screen all 2,913 raw records with the benchmark prompt.")
    parser.add_argument("--restart", action="store_true", help="Discard the full-corpus checkpoint and restart.")
    parser.add_argument(
        "--reasoning-effort", choices=benchmark.SUPPORTED_REASONING_EFFORTS,
        default=benchmark.DEFAULT_REASONING_EFFORT,
    )
    args = parser.parse_args(argv)
    base_directory = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=base_directory / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: missing required environment variable: OPENAI_API_KEY", file=sys.stderr)
        return 2
    try:
        execute(base_directory, reasoning_effort=args.reasoning_effort, restart=args.restart)
    except (benchmark.PipelineError, OSError, ValueError, openai.OpenAIError) as error:
        print(f"Error: {error}", file=sys.stderr)
        print(f"Successful records remain in {CHECKPOINT_FILENAME}; rerun to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
