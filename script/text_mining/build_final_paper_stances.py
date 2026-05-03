#!/usr/bin/env python3
"""
Build final paper-level stances by backfilling unclear traditional outputs
with downstream classifier predictions.
"""

from __future__ import annotations

import argparse
import csv


def require_columns(fieldnames: list[str], required: list[str], csv_path: str) -> None:
    missing = [col for col in required if col not in fieldnames]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")


def load_csv_rows(path: str, required_columns: list[str]) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames or [], required_columns, path)
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill unclear traditional labels with classifier predictions")
    parser.add_argument("--traditional", default="output_csvs/traditional_stance_labels.csv")
    parser.add_argument("--classifier", default="output_csvs/bow_stance_predictions.csv")
    parser.add_argument("--output", default="output_csvs/final_paper_stances.csv")
    args = parser.parse_args()

    traditional_rows = load_csv_rows(
        args.traditional,
        ["PaperID", "PredictedStance", "ConfidenceScore", "ConfidenceMargin", "EvidenceSentences"],
    )
    classifier_rows = load_csv_rows(
        args.classifier,
        ["PaperID", "DOI", "Biomarkers", "Drugs", "PredictedStance", "ConfidenceScore", "ConfidenceMargin"],
    )

    traditional_by_paper = {row["PaperID"]: row for row in traditional_rows}

    final_rows = []
    source_counts = {"traditional_non_unclear": 0, "classifier_backfill_unclear": 0, "classifier_only": 0}

    for pred in classifier_rows:
        paper_id = pred["PaperID"]
        classifier_label = pred["PredictedStance"].strip().lower()
        classifier_conf = float(pred["ConfidenceScore"])
        classifier_margin = float(pred["ConfidenceMargin"])
        traditional = traditional_by_paper.get(paper_id)

        if traditional is None:
            final_label = classifier_label
            final_conf = classifier_conf
            final_margin = classifier_margin
            final_source = "classifier_only"
            traditional_label = ""
            traditional_conf = ""
            traditional_margin = ""
            evidence = ""
            source_counts[final_source] += 1
        else:
            traditional_label = traditional["PredictedStance"].strip().lower()
            traditional_conf = float(traditional["ConfidenceScore"])
            traditional_margin = float(traditional["ConfidenceMargin"])
            evidence = traditional["EvidenceSentences"]

            if traditional_label != "unclear":
                final_label = traditional_label
                final_conf = traditional_conf
                final_margin = traditional_margin
                final_source = "traditional_non_unclear"
                source_counts[final_source] += 1
            else:
                final_label = classifier_label
                final_conf = classifier_conf
                final_margin = classifier_margin
                final_source = "classifier_backfill_unclear"
                source_counts[final_source] += 1

        final_rows.append(
            {
                "PaperID": paper_id,
                "DOI": pred["DOI"],
                "Biomarkers": pred["Biomarkers"],
                "Drugs": pred["Drugs"],
                "TraditionalLabel": traditional_label,
                "TraditionalConfidence": f"{traditional_conf:.6f}" if traditional_conf != "" else "",
                "TraditionalMargin": f"{traditional_margin:.6f}" if traditional_margin != "" else "",
                "ClassifierLabel": classifier_label,
                "ClassifierConfidence": f"{classifier_conf:.6f}",
                "ClassifierMargin": f"{classifier_margin:.6f}",
                "FinalLabel": final_label,
                "FinalConfidence": f"{final_conf:.6f}",
                "FinalMargin": f"{final_margin:.6f}",
                "FinalSource": final_source,
                "TraditionalEvidenceSentences": evidence,
            }
        )

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "PaperID",
                "DOI",
                "Biomarkers",
                "Drugs",
                "TraditionalLabel",
                "TraditionalConfidence",
                "TraditionalMargin",
                "ClassifierLabel",
                "ClassifierConfidence",
                "ClassifierMargin",
                "FinalLabel",
                "FinalConfidence",
                "FinalMargin",
                "FinalSource",
                "TraditionalEvidenceSentences",
            ],
        )
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Final rows: {len(final_rows)}")
    print(f"Traditional non-unclear kept: {source_counts['traditional_non_unclear']}")
    print(f"Traditional unclear backfilled by classifier: {source_counts['classifier_backfill_unclear']}")
    print(f"Classifier-only rows (no traditional row): {source_counts['classifier_only']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
