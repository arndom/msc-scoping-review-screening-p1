# OSS Abandonment Screening Pipeline

This directory contains the reproducible sampling and LLM-assisted title-and-abstract screening stages for a PRISMA-ScR compliant scoping review on **Understanding Abandonment in Open Source Software**.

## Requirements

- Python 3.9
- Dependencies pinned in `requirements.txt`
- An OpenAI API key in `.env` for the benchmarking stage

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create or update `.env` in this directory:

```dotenv
OPENAI_API_KEY="your-api-key"
```

The pipeline loads this file with `python-dotenv`. A key already set in the process environment takes precedence and is not overwritten. `.env` is excluded by `.gitignore`; never commit it or place the key directly in source code. The model is fixed as `gpt-6-luna`, so `OPENAI_MODEL` is not required.

## Stage 1: reproducible sampling

`create_ground_truth_sample.py`:

- Reads `raw_records_2913.csv` from the script directory.
- Validates that the input contains exactly 2,913 records and all required source columns.
- Selects exactly 291 records without replacement using pandas with random seed `42`.
- Preserves the seeded random order and generates `S/N` values from 1 through 291.
- Writes `ground_truth_sample_291.csv` using UTF-8, LF line endings, and no pandas index.
- Keeps records with blank abstracts eligible for sampling.

Run it with:

```powershell
python create_ground_truth_sample.py
```

The generated sample contains:

```text
S/N, Item Type, Title, Abstract Note, DOI, Url, Language
```

## Manual screening data

`ground_truth_screened_291.csv` is the manually labelled benchmark reference. It contains the sampled source fields plus:

- `Decision`
- `Exclusion Code`
- `Notes`

The benchmark script validates that its 291 source records match `ground_truth_sample_291.csv`. Human decisions, exclusion codes, and notes are never sent to the API.

## Stage 2: LLM screening and benchmarking

`benchmark_pipeline.py` sends one sampled record per request through the OpenAI Responses API using Pydantic Structured Outputs. It sends `Title` and `Abstract Note` as primary evidence and includes `Item Type`, `DOI`, `Url`, and `Language` as secondary metadata.

The model applies the protocol-derived exclusions in strict hierarchical order:

1. `EXC-1: Non-OSS Context`
2. `EXC-2: Out of Scope Concept`
3. `EXC-3: Non-SE Context`
4. `EXC-4: Grey / Non Peer Reviewed`
5. `EXC-5: Secondary Study`

The first explicitly supported exclusion is selected. In particular, an explicitly non-OSS population receives EXC-1 even if a later exclusion could also describe it. Missing or ambiguous evidence is assigned `Unsure`, not implicitly excluded. EXC-6 and EXC-7 are reserved for full-text screening and disabled here.

The structured result adds:

- `llm_decision`
- `llm_exclusion_code`
- `llm_notes`

The default run uses reasoning effort `none`:

```powershell
python benchmark_pipeline.py
```

To test Luna with `low` reasoning:

```powershell
python benchmark_pipeline.py --reasoning-effort low --restart
```

Changing reasoning effort changes the checkpoint signature. Use `--restart` to start the new experimental configuration rather than mixing results from two configurations. Because each completed run uses the same output filenames, preserve the `none` results before starting the `low` run if you want to compare them:

```powershell
Copy-Item benchmark_results.json benchmark_results_none.json
Copy-Item llm_screened_291.csv llm_screened_291_none.csv
```

### Temperature and reasoning effort

Temperature controls variation during token selection. Lower values concentrate selection on the model's most probable output; higher values allow more varied responses. `temperature=0.0` is appropriate for this benchmark because it reduces unnecessary output variation, but it does not mathematically guarantee identical API results across repeated runs.

Reasoning effort controls a separate mechanism: how many additional, non-visible reasoning tokens the model may use before returning its answer. `none` does not mean that the model cannot read or classify a record; it disables the additional reasoning-token stage. `low` permits a modest amount of extra deliberation that may help with the hierarchical exclusion rules.

For GPT-6 Luna, this pipeline uses these valid combinations:

| CLI setting | API reasoning effort | Temperature |
| --- | --- | --- |
| default or `--reasoning-effort none` | `none` | `0.0` |
| `--reasoning-effort low` | `low` | omitted |

OpenAI requires temperature to be omitted when GPT-6 reasoning effort is not `none`. See the [GPT-6 Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna) and [OpenAI API deployment guidance](https://developers.openai.com/api/docs/guides/deployment-checklist).

### Resuming and restarting

After each successful API response, the pipeline appends a record to `.benchmark_checkpoint.jsonl`. If a transient API error interrupts the run, rerun the same command to resume completed work.

Use `--restart` when intentionally starting over or changing the reasoning configuration:

```powershell
python benchmark_pipeline.py --restart
```

The checkpoint signature covers the model, reasoning effort, effective temperature, prompt, output schema, sample, and protocol. An incompatible checkpoint is rejected rather than silently reused.

### Outputs and metrics

The completed run writes:

- `llm_screened_291.csv`: source fields plus the three LLM result columns.
- `benchmark_results.json`: prompt and input hashes, model and reasoning configuration, dependencies, timing, token usage, class counts, metrics, and error record IDs.

## Full-corpus screening

`screen_full_corpus.py` applies the exact same prompt, Pydantic schema, model settings, and one-record-per-request classifier used by `benchmark_pipeline.py` to all 2,913 records in `raw_records_2913.csv`. It generates sequential `S/N` identifiers. The output CSV contains only these columns, in order: `S/N`, `Item Type`, `Publication Year`, `Author`, `Title`, `DOI`, `Url`, `Abstract Note`, `Decision`, `Exclusion Code`, and `Notes`. Only the benchmark's screening fields (`S/N`, `Item Type`, `Title`, `Abstract Note`, `DOI`, `Url`, and `Language`) are sent to the model; the other output metadata fields are retained for your screening sheet but not sent.

Run the full screening with the default `none` reasoning effort:

```powershell
python screen_full_corpus.py
```

Or select `low` reasoning effort:

```powershell
python screen_full_corpus.py --reasoning-effort low
```

The script writes `llm_screened_2913.csv` and `llm_screening_report_2913.json`. The JSON report includes the exact prompt and version/hash, input and protocol hashes, requested/returned model identifiers, run settings, token usage, timing, decision counts and percentages (Exclude, Include, Unsure, and Include-or-Unsure combined), and counts for each exclusion code (including records with no exclusion code). Percentages use all screened records as the denominator. There is no human-label metric calculation for this full-corpus run because no human-screened ground truth is supplied for all 2,913 records.

Successful record results are checkpointed in `.full_corpus_checkpoint.jsonl` and reused when rerunning with the same input, prompt, model, and reasoning configuration. Use `--restart` to deliberately discard that checkpoint and classify every record again. Changing the reasoning effort requires a separate checkpoint signature; preserve prior outputs before rerunning if you want to retain both configurations.

`Include` and `Unsure` form the positive Candidate Set. The report calculates:

- Candidate-set recall/sensitivity, with a target of at least 0.95.
- False-negative rate.
- Precision.
- Specificity.
- Binary candidate-set accuracy.
- Three-class decision accuracy.
- Exclusion-code accuracy across human-excluded records.

## Offline tests

The test suite uses a mocked OpenAI client and does not spend API credits:

```powershell
python -m unittest -v
```

It covers structured-result invariants, exclusion hierarchy, CSV validation, API payload isolation, transient retries, refusals, reasoning/temperature compatibility, checkpoint validation, metrics, and deterministic JSON serialization.
