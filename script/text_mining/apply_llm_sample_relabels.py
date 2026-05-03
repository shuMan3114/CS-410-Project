#!/usr/bin/env python3
"""
Apply LLM-reviewed stance relabel fixes to sampled evaluation rows.

This is intentionally explicit and strict:
- No silent fallback behavior.
- Raises if required columns are missing.
- Raises if a configured PaperID is not found.
"""

from __future__ import annotations

import argparse
import csv


FIXES = [
    {
        "PaperID": "39058425",
        "old_label": "supportive",
        "llm_label": "opposing",
        "rationale_short": "Evidence says adding ipatasertib did not improve efficacy.",
    },
    {
        "PaperID": "10424760",
        "old_label": "supportive",
        "llm_label": "opposing",
        "rationale_short": "AKT1-overexpressing cells were less sensitive to wortmannin.",
    },
    {
        "PaperID": "41202161",
        "old_label": "neutral",
        "llm_label": "supportive",
        "rationale_short": "Praluzatamab ravtansine delayed tumor growth in PDX models.",
    },
    {
        "PaperID": "21256123",
        "old_label": "supportive",
        "llm_label": "opposing",
        "rationale_short": "ALDH1A1 knockdown increased acrolein sensitivity (native ALDH1A1 implies resistance).",
    },
    {
        "PaperID": "11320670",
        "old_label": "supportive",
        "llm_label": "opposing",
        "rationale_short": "Ifosfamide sensitivity is inverse to ALDH levels (resistance signal).",
    },
    {
        "PaperID": "40467521",
        "old_label": "opposing",
        "llm_label": "supportive",
        "rationale_short": "Evinacumab substantially reduced LDL-C and TG levels.",
    },
    {
        "PaperID": "40024348",
        "old_label": "opposing",
        "llm_label": "supportive",
        "rationale_short": "Basroparib is described as treating MEK inhibitor-resistant CRC.",
    },
    {
        "PaperID": "37146911",
        "old_label": "supportive",
        "llm_label": "opposing",
        "rationale_short": "KRAS/APC co-mutated cohort did not benefit from bevacizumab.",
    },
    {
        "PaperID": "40084594",
        "old_label": "opposing",
        "llm_label": "supportive",
        "rationale_short": "Inclisiran lowered lipid markers without serious safety concerns.",
    },
    {
        "PaperID": "26572704",
        "old_label": "opposing",
        "llm_label": "supportive",
        "rationale_short": "Sorafenib and GSK126 inhibited proliferation in mutation-defined tumors.",
    },
    {
        "PaperID": "26639724",
        "old_label": "neutral",
        "llm_label": "opposing",
        "rationale_short": "Variant causes prolonged neuromuscular block after mivacurium (adverse response).",
    },
    {
        "PaperID": "38723373",
        "old_label": "neutral",
        "llm_label": "supportive",
        "rationale_short": "Improved PFS/OS vs vemurafenib indicates efficacy benefit.",
    },
]

LABEL_COLUMNS_TO_UPDATE = ["sampling_label", "phase_1_label", "phase_2_label"]


def require_columns(fieldnames: list[str], required: list[str], csv_path: str) -> None:
    missing = [col for col in required if col not in fieldnames]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply LLM-reviewed relabel fixes to sample evaluation CSV")
    parser.add_argument("--input-csv", default="output_csvs/stance_labels_eval_sample.csv")
    parser.add_argument("--output-csv", default="output_csvs/stance_labels_eval_sample.csv")
    parser.add_argument("--flags-output", default="output_csvs/stance_labels_eval_sample_llm_flags.csv")
    parser.add_argument("--max-paper-fixes", type=int, default=0, help="0 means apply all configured fixes")
    args = parser.parse_args()

    with open(args.input_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        require_columns(
            fieldnames,
            ["PaperID", "LLMReviewLabel", "ManualNotes"] + LABEL_COLUMNS_TO_UPDATE,
            args.input_csv,
        )
        rows = list(reader)

    fixes = FIXES
    if args.max_paper_fixes > 0:
        fixes = fixes[: args.max_paper_fixes]

    rows_by_paper = {row["PaperID"]: row for row in rows}
    flag_rows = []
    for fix in fixes:
        paper_id = fix["PaperID"]
        if paper_id not in rows_by_paper:
            raise ValueError(f"Configured fix PaperID not found in CSV: {paper_id}")
        row = rows_by_paper[paper_id]

        for col in LABEL_COLUMNS_TO_UPDATE:
            if row[col].strip().lower() == fix["old_label"]:
                row[col] = fix["llm_label"]

        row["LLMReviewLabel"] = fix["llm_label"]
        existing_notes = row["ManualNotes"].strip()
        flag_note = (
            f"FLAG_RELABELED phase_2_label {fix['old_label']}->{fix['llm_label']}; "
            f"Reason: {fix['rationale_short']}"
        )
        if flag_note in existing_notes:
            row["ManualNotes"] = existing_notes
        else:
            row["ManualNotes"] = f"{existing_notes} | {flag_note}".strip(" |")
        flag_rows.append(
            {
                "PaperID": paper_id,
                "OldLabel": fix["old_label"],
                "NewLabel": fix["llm_label"],
                "Flag": "FLAG_RELABELED",
                "Reason": fix["rationale_short"],
            }
        )

    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(args.flags_output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["PaperID", "OldLabel", "NewLabel", "Flag", "Reason"],
        )
        writer.writeheader()
        writer.writerows(flag_rows)

    print(f"Applied relabel fixes: {len(fixes)}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Flag CSV: {args.flags_output}")


if __name__ == "__main__":
    main()
