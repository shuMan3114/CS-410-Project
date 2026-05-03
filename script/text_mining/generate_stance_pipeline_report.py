#!/usr/bin/env python3
"""
Generate a Markdown report for the stance-modeling pipeline.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter


def require_columns(fieldnames: list[str], required: list[str], csv_path: str) -> None:
    missing = [col for col in required if col not in fieldnames]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")


def read_rows(path: str, required: list[str]) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames or [], required, path)
        return list(reader)


def pct(count: int, total: int) -> str:
    return f"{(100.0 * count / total):.2f}%" if total else "0.00%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stance pipeline report markdown")
    parser.add_argument("--traditional", default="output_csvs/traditional_stance_labels.csv")
    parser.add_argument("--audit-summary", default="output_csvs/traditional_label_audit_summary.json")
    parser.add_argument("--bow-eval", default="output_csvs/bow_stance_eval.json")
    parser.add_argument("--final-stances", default="output_csvs/final_paper_stances.csv")
    parser.add_argument("--output-md", default="doc/CS_410_Stance_Pipeline_Report.md")
    args = parser.parse_args()

    traditional_rows = read_rows(
        args.traditional,
        ["PaperID", "PredictedStance", "ConfidenceScore", "P_supportive", "P_opposing", "P_neutral"],
    )
    final_rows = read_rows(args.final_stances, ["PaperID", "FinalLabel", "FinalSource"])

    with open(args.audit_summary, encoding="utf-8") as handle:
        audit = json.load(handle)
    with open(args.bow_eval, encoding="utf-8") as handle:
        bow_eval = json.load(handle)

    traditional_counts = Counter(row["PredictedStance"].strip().lower() for row in traditional_rows)
    final_counts = Counter(row["FinalLabel"].strip().lower() for row in final_rows)
    final_sources = Counter(row["FinalSource"].strip() for row in final_rows)

    traditional_total = len(traditional_rows)
    final_total = len(final_rows)
    today = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# CS-410 Stance Modeling Pipeline Report

Generated: {today}

## 1) Pipeline Overview

The current stance pipeline runs in three stages:

1. **Traditional stance mining (`traditional_stance_mining.py`)**
   - Extracts gene-drug co-mentioned sentence segments.
   - Mines stance patterns with PMI.
   - Scores papers by class and converts to class probabilities (`P_supportive`, `P_opposing`, `P_neutral`).
   - Produces `traditional_stance_labels.csv`.

2. **Downstream BoW classifier (`train_bow_stance_classifier.py`)**
   - Uses non-unclear traditional labels as weak supervision.
   - Trains multinomial Naive Bayes on abstract text.
   - Produces `bow_stance_predictions.csv` for all papers.

3. **Final stance consolidation (`build_final_paper_stances.py`)**
   - Keeps traditional non-unclear labels.
   - Replaces traditional `unclear` with downstream classifier predictions.
   - Uses classifier-only labels for papers without a traditional row.
   - Produces `final_paper_stances.csv`.

## 2) Traditional Method Outputs

- Total traditional rows: **{traditional_total}**
- Traditional label distribution:
  - supportive: {traditional_counts.get("supportive", 0)} ({pct(traditional_counts.get("supportive", 0), traditional_total)})
  - opposing: {traditional_counts.get("opposing", 0)} ({pct(traditional_counts.get("opposing", 0), traditional_total)})
  - neutral: {traditional_counts.get("neutral", 0)} ({pct(traditional_counts.get("neutral", 0), traditional_total)})
  - unclear: {traditional_counts.get("unclear", 0)} ({pct(traditional_counts.get("unclear", 0), traditional_total)})

## 3) Traditional Label Audit (All Training-Eligible Labels)

Audit source: `traditional_label_audit_summary.json` from `audit_traditional_labels.py`.

- Training-eligible labeled rows checked: **{audit["training_rows_checked"]}**
- Auditable rows (clear rubric decision): **{audit["auditable_rows"]}**
- Unresolved rows (insufficient/conflicting evidence cues): **{audit["unresolved_rows"]}**
- Mismatch rows: **{audit["mismatch_rows"]}**
- **Mean mistake (auditable denominator)**: **{audit["mean_mistake"]:.6f}** ({audit["mean_mistake"] * 100:.2f}%)
- **Mean mistake (all-labeled denominator)**: **{audit["mean_mistake_all_labeled"]:.6f}** ({audit["mean_mistake_all_labeled"] * 100:.2f}%)

Interpretation for reporting:
- If you want a strict “checked-only” error rate, use **{audit["mean_mistake"] * 100:.2f}%**.
- If you want a conservative full-set rate, use **{audit["mean_mistake_all_labeled"] * 100:.2f}%**.

## 4) Downstream Classifier Performance

From `bow_stance_eval.json`:

- Train accuracy: **{bow_eval["train"]["accuracy"]:.6f}** on {bow_eval["train"]["count"]} rows
- Validation accuracy: **{bow_eval["validation"]["accuracy"]:.6f}** on {bow_eval["validation"]["count"]} rows

## 5) Final Paper Stances (After Unclear Backfill)

- Total final rows: **{final_total}**
- Final source composition:
  - `traditional_non_unclear`: {final_sources.get("traditional_non_unclear", 0)} ({pct(final_sources.get("traditional_non_unclear", 0), final_total)})
  - `classifier_backfill_unclear`: {final_sources.get("classifier_backfill_unclear", 0)} ({pct(final_sources.get("classifier_backfill_unclear", 0), final_total)})
  - `classifier_only`: {final_sources.get("classifier_only", 0)} ({pct(final_sources.get("classifier_only", 0), final_total)})

- Final label distribution:
  - supportive: {final_counts.get("supportive", 0)} ({pct(final_counts.get("supportive", 0), final_total)})
  - opposing: {final_counts.get("opposing", 0)} ({pct(final_counts.get("opposing", 0), final_total)})
  - neutral: {final_counts.get("neutral", 0)} ({pct(final_counts.get("neutral", 0), final_total)})

## 6) Key Output Files

- `output_csvs/traditional_stance_labels.csv`
- `output_csvs/traditional_label_audit.csv`
- `output_csvs/traditional_label_audit_summary.json`
- `output_csvs/bow_stance_predictions.csv`
- `output_csvs/bow_stance_eval.json`
- `output_csvs/final_paper_stances.csv`
"""

    with open(args.output_md, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(args.output_md)


if __name__ == "__main__":
    main()
