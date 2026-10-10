"""Create the reproducible title-and-abstract ground-truth sample."""

from pathlib import Path
import sys

import pandas as pd


INPUT_FILENAME = "raw_records_2913.csv"
OUTPUT_FILENAME = "ground_truth_sample_291.csv"
EXPECTED_INPUT_RECORDS = 2_913
SAMPLE_SIZE = 291
RANDOM_SEED = 42
SOURCE_COLUMNS = ["Item Type", "Title", "Abstract Note", "DOI", "Url", "Language"]
OUTPUT_COLUMNS = ["S/N", *SOURCE_COLUMNS]


def create_ground_truth_sample(input_path: Path, output_path: Path) -> None:
    """Validate the source data and write the deterministic random sample."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    records = pd.read_csv(input_path, encoding="utf-8-sig", keep_default_na=False)

    missing_columns = [column for column in SOURCE_COLUMNS if column not in records.columns]
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required column(s): " + ", ".join(missing_columns)
        )

    if len(records) != EXPECTED_INPUT_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_INPUT_RECORDS} input records, found {len(records)}."
        )

    sample = records.sample(
        n=SAMPLE_SIZE,
        replace=False,
        random_state=RANDOM_SEED,
    )
    if not sample.index.is_unique:
        raise RuntimeError("Sampling unexpectedly selected a record more than once.")

    output = sample.loc[:, SOURCE_COLUMNS].copy()
    output.insert(0, "S/N", range(1, SAMPLE_SIZE + 1))

    if list(output.columns) != OUTPUT_COLUMNS or len(output) != SAMPLE_SIZE:
        raise RuntimeError("Generated sample failed its shape or column validation.")

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def main() -> int:
    base_directory = Path(__file__).resolve().parent
    input_path = base_directory / INPUT_FILENAME
    output_path = base_directory / OUTPUT_FILENAME

    try:
        create_ground_truth_sample(input_path, output_path)
    except (FileNotFoundError, ValueError, RuntimeError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Created {output_path.name} with {SAMPLE_SIZE} records "
        f"using random seed {RANDOM_SEED}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
