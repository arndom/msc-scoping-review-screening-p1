"""Merge selected ground-truth columns into wip.csv using S/N."""

from pathlib import Path
import sys

import pandas as pd


GROUND_TRUTH_FILENAME = "ground_truth_sample_291.csv"
WIP_FILENAME = "wip.csv"
KEY_COLUMN = "S/N"
SOURCE_COLUMNS = ["Item Type", "Title", "Abstract Note", "DOI", "Url", "Language"]


def merge_source_columns(ground_truth_path: Path, wip_path: Path) -> None:
    """Update the selected columns in wip_path while preserving all other columns."""
    for path in (ground_truth_path, wip_path):
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

    ground_truth = pd.read_csv(
        ground_truth_path, encoding="utf-8-sig", keep_default_na=False
    )
    wip = pd.read_csv(wip_path, encoding="utf-8-sig", keep_default_na=False)

    required_ground_truth_columns = [KEY_COLUMN, *SOURCE_COLUMNS]
    missing = [
        column
        for column in required_ground_truth_columns
        if column not in ground_truth.columns
    ]
    if missing:
        raise ValueError(
            "Ground-truth CSV is missing required column(s): " + ", ".join(missing)
        )
    if KEY_COLUMN not in wip.columns:
        raise ValueError(f"WIP CSV is missing the key column: {KEY_COLUMN}")
    if ground_truth[KEY_COLUMN].duplicated().any() or wip[KEY_COLUMN].duplicated().any():
        raise ValueError(f"Both CSV files must contain unique {KEY_COLUMN} values.")

    ground_truth_keys = set(ground_truth[KEY_COLUMN])
    wip_keys = set(wip[KEY_COLUMN])
    if ground_truth_keys != wip_keys:
        missing_from_wip = ground_truth_keys - wip_keys
        missing_from_ground_truth = wip_keys - ground_truth_keys
        raise ValueError(
            f"{KEY_COLUMN} values do not match: "
            f"{len(missing_from_wip)} missing from WIP and "
            f"{len(missing_from_ground_truth)} missing from ground truth."
        )

    source_by_key = ground_truth.set_index(KEY_COLUMN)
    merged = wip.copy()
    for column in SOURCE_COLUMNS:
        merged[column] = merged[KEY_COLUMN].map(source_by_key[column])

    other_columns = [
        column
        for column in wip.columns
        if column != KEY_COLUMN and column not in SOURCE_COLUMNS
    ]
    merged = merged.loc[:, [KEY_COLUMN, *SOURCE_COLUMNS, *other_columns]]
    merged.to_csv(wip_path, index=False, encoding="utf-8", lineterminator="\n")


def main() -> int:
    base_directory = Path(__file__).resolve().parent
    ground_truth_path = base_directory / GROUND_TRUTH_FILENAME
    wip_path = base_directory / WIP_FILENAME

    try:
        merge_source_columns(ground_truth_path, wip_path)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Merged {', '.join(SOURCE_COLUMNS)} into {wip_path.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
