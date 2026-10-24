"""Offline checks for the full-corpus runner."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

import benchmark_pipeline as benchmark
import screen_full_corpus as full


class FullCorpusTests(unittest.TestCase):
    def test_input_validation_and_sequential_serials(self):
        rows = {
            "Item Type": ["journalArticle"] * full.EXPECTED_RECORDS,
            "Title": [f"Title {number}" for number in range(full.EXPECTED_RECORDS)],
            "Abstract Note": [""] * full.EXPECTED_RECORDS,
            "DOI": [""] * full.EXPECTED_RECORDS,
            "Url": [""] * full.EXPECTED_RECORDS,
            "Language": [""] * full.EXPECTED_RECORDS,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / full.INPUT_FILENAME
            pd.DataFrame(rows).to_csv(path, index=False)
            loaded = full.read_input(path)
        self.assertEqual(len(loaded), 2913)
        self.assertEqual(loaded["S/N"].tolist(), list(range(1, 2914)))
        self.assertEqual(loaded.loc[0, "Abstract Note"], "")

    def test_count_results_includes_all_decisions_and_empty_codes(self):
        data = pd.DataFrame(
            {
                "Decision": ["Include", "Unsure", "Exclude", "Exclude"],
                "Exclusion Code": [
                    "",
                    "",
                    benchmark.ExclusionCode.NON_OSS.value,
                    benchmark.ExclusionCode.OUT_OF_SCOPE.value,
                ],
            }
        )
        decisions, codes = full.count_results(data)
        self.assertEqual(decisions, {"Include": 1, "Unsure": 1, "Exclude": 2})
        self.assertEqual(codes[benchmark.ExclusionCode.NON_OSS.value], 1)
        self.assertEqual(codes[benchmark.ExclusionCode.OUT_OF_SCOPE.value], 1)
        self.assertEqual(codes["No exclusion code"], 2)

    def test_decision_percentages_include_combined_and_individual_categories(self):
        percentages = full.decision_percentages(
            {"Include": 100, "Unsure": 50, "Exclude": 150}, 300
        )
        self.assertEqual(percentages["Exclude"]["percentage"], 50.0)
        self.assertEqual(percentages["Include"]["percentage"], 33.33)
        self.assertEqual(percentages["Unsure"]["percentage"], 16.67)
        self.assertEqual(percentages["Include_or_Unsure"]["percentage"], 50.0)

    def test_output_uses_phase_sheet_column_order_and_names(self):
        raw = pd.DataFrame(
            {
                "S/N": range(1, full.EXPECTED_RECORDS + 1),
                "Item Type": ["journalArticle"] * full.EXPECTED_RECORDS,
                "Publication Year": [2024] * full.EXPECTED_RECORDS,
                "Author": ["Author"] * full.EXPECTED_RECORDS,
                "Title": ["Title"] * full.EXPECTED_RECORDS,
                "DOI": [""] * full.EXPECTED_RECORDS,
                "Url": [""] * full.EXPECTED_RECORDS,
                "Abstract Note": [""] * full.EXPECTED_RECORDS,
                "Notes": ["Raw bibliographic note"] * full.EXPECTED_RECORDS,
                "Extra field": ["not included"] * full.EXPECTED_RECORDS,
            }
        )
        predictions = pd.DataFrame(
            {
                "S/N": range(1, full.EXPECTED_RECORDS + 1),
                "llm_decision": ["Include"] * full.EXPECTED_RECORDS,
                "llm_exclusion_code": [None] * full.EXPECTED_RECORDS,
                "llm_notes": ["LLM rationale"] * full.EXPECTED_RECORDS,
            }
        )
        output = full.build_output(raw, predictions)
        self.assertEqual(output.columns.tolist(), full.OUTPUT_COLUMNS)
        self.assertNotIn("Extra field", output.columns)
        self.assertEqual(output.columns.tolist().count("Notes"), 1)
        self.assertEqual(output.loc[0, "Notes"], "LLM rationale")
        self.assertEqual(output.loc[0, "Decision"], "Include")


if __name__ == "__main__":
    unittest.main()
