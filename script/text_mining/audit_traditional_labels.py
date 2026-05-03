#!/usr/bin/env python3
"""
Audit traditional stance labels with an evidence-text rubric.

This script is intentionally strict:
- It raises on missing columns.
- It does not use fallback defaults.
- It outputs explicit mismatch flags and summary metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter


SUPPORTED_LABELS = {"supportive", "opposing", "neutral"}

# Lexical rubric for evidence-driven stance audit.
SUPPORTIVE_PATTERNS = [
    r"\bimprov(?:e|ed|es|ement)\b",
    r"\bbenefit(?:ed|s)?\b",
    r"\bbeneficial\b",
    r"\beffective\b",
    r"\befficacy\b",
    r"\brespond(?:ed|er|ers)?\b",
    r"\bsensitiv(?:e|ity)\b",
    r"\bsynerg(?:y|istic)\b",
    r"\binhibit(?:ed|ion|s)?\b",
    r"\breduc(?:e|ed|es|ing)\b",
    r"\bdecreas(?:e|ed|es|ing)\b",
    r"\bimproved progression-free survival\b",
    r"\bimproved overall survival\b",
    r"\bdelayed tumor growth\b",
    r"\bwithout increasing the risk of serious adverse events\b",
]
OPPOSING_PATTERNS = [
    r"\bresistan(?:t|ce)\b",
    r"\bdrug-resistant\b",
    r"\bimatinib-resistant\b",
    r"\btamoxifen-resistant\b",
    r"\bineffective\b",
    r"\bno benefit\b",
    r"\bdid not benefit\b",
    r"\black of benefit\b",
    r"\bdid not improve\b",
    r"\bfailed\b",
    r"\bfailure\b",
    r"\badverse\b",
    r"\btoxic\b",
    r"\bprogression\b",
    r"\bpoor progression-free survival\b",
    r"\bless sensitive\b",
    r"\bcontraindicat(?:e|ed)\b",
    r"\bprolonged neuromuscular block\b",
]
NEUTRAL_PATTERNS = [
    r"\bassociat(?:ed|ion)\b",
    r"\bcorrelat(?:ed|ion)\b",
    r"\binvestigat(?:ed|e|ing)\b",
    r"\bevaluat(?:ed|e|ing)\b",
    r"\breview\b",
    r"\bobjective\b",
    r"\bbackground\b",
    r"\bpreliminary\b",
    r"\bunclear\b",
    r"\buncertain\b",
]


def require_columns(fieldnames: list[str], required: list[str], csv_path: str) -> None:
    missing = [col for col in required if col not in fieldnames]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")


def regex_count(text: str, patterns: list[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text))


def audit_label_from_evidence(evidence: str) -> tuple[str, str]:
    """
    Returns: (audit_label, rationale_short)
    """
    text = evidence.strip().lower()
    if not text:
        return "unclear", "No evidence sentence available."

    supportive_score = regex_count(text, SUPPORTIVE_PATTERNS)
    opposing_score = regex_count(text, OPPOSING_PATTERNS)
    neutral_score = regex_count(text, NEUTRAL_PATTERNS)

    # Negation-focused opposing cues.
    if re.search(r"\b(no|not|did not|without)\s+(improv(?:e|ed|ement)|benefit|effective|efficacy|response)\b", text):
        opposing_score += 2
    # "less sensitive" / "more sensitive" directional cues.
    if re.search(r"\bless sensitive\b", text):
        opposing_score += 2
    if re.search(r"\bmore sensitive\b", text):
        supportive_score += 2

    ordered = sorted(
        [
            ("supportive", supportive_score),
            ("opposing", opposing_score),
            ("neutral", neutral_score),
        ],
        key=lambda pair: pair[1],
        reverse=True,
    )
    top_label, top_score = ordered[0]
    second_score = ordered[1][1]
    if top_score == 0:
        return "unclear", "No directional cues detected in evidence."
    if top_score == second_score:
        return "unclear", "Conflicting cues with no dominant stance signal."

    rationale = (
        f"supportive={supportive_score}, opposing={opposing_score}, neutral={neutral_score}; "
        f"dominant={top_label}"
    )
    return top_label, rationale


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit traditional stance labels via evidence rubric")
    parser.add_argument("--input-csv", default="output_csvs/traditional_stance_labels.csv")
    parser.add_argument("--output-csv", default="output_csvs/traditional_label_audit.csv")
    parser.add_argument("--summary-json", default="output_csvs/traditional_label_audit_summary.json")
    parser.add_argument("--label-col", default="PredictedStance")
    parser.add_argument("--confidence-col", default="ConfidenceScore")
    parser.add_argument("--evidence-col", default="EvidenceSentences")
    parser.add_argument("--min-confidence", type=float, default=0.60)
    args = parser.parse_args()

    with open(args.input_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        require_columns(
            fieldnames,
            ["PaperID", args.label_col, args.confidence_col, args.evidence_col],
            args.input_csv,
        )
        rows = list(reader)

    audited_rows = []
    label_distribution = Counter()
    audit_distribution = Counter()
    mismatch_distribution = Counter()

    training_rows = 0
    auditable_rows = 0
    mismatch_rows = 0

    for row in rows:
        label = row[args.label_col].strip().lower()
        confidence = float(row[args.confidence_col])

        if label not in SUPPORTED_LABELS:
            continue
        if confidence < args.min_confidence:
            continue

        training_rows += 1
        label_distribution[label] += 1

        audit_label, rationale = audit_label_from_evidence(row[args.evidence_col])
        audit_distribution[audit_label] += 1
        is_auditable = audit_label in SUPPORTED_LABELS
        if is_auditable:
            auditable_rows += 1

        is_mismatch = bool(is_auditable and audit_label != label)
        if is_mismatch:
            mismatch_rows += 1
            mismatch_distribution[f"{label}->{audit_label}"] += 1

        audited_rows.append(
            {
                "PaperID": row["PaperID"],
                "TraditionalLabel": label,
                "TraditionalConfidence": f"{confidence:.6f}",
                "AuditLabel": audit_label,
                "MismatchFlag": "1" if is_mismatch else "0",
                "AuditRationale": rationale,
                "EvidenceSentences": row[args.evidence_col],
            }
        )

    mean_mistake = (mismatch_rows / auditable_rows) if auditable_rows else 0.0
    mean_mistake_all_labeled = (mismatch_rows / training_rows) if training_rows else 0.0
    unresolved_rows = training_rows - auditable_rows
    unresolved_rate = (unresolved_rows / training_rows) if training_rows else 0.0
    summary = {
        "training_rows_checked": training_rows,
        "auditable_rows": auditable_rows,
        "unresolved_rows": unresolved_rows,
        "unresolved_rate": unresolved_rate,
        "mismatch_rows": mismatch_rows,
        "mean_mistake": mean_mistake,
        "mean_mistake_all_labeled": mean_mistake_all_labeled,
        "traditional_label_distribution": dict(label_distribution),
        "audit_label_distribution": dict(audit_distribution),
        "mismatch_distribution": dict(mismatch_distribution),
        "notes": "Mean mistake is mismatch_rows / auditable_rows under evidence-text rubric.",
    }

    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "PaperID",
                "TraditionalLabel",
                "TraditionalConfidence",
                "AuditLabel",
                "MismatchFlag",
                "AuditRationale",
                "EvidenceSentences",
            ],
        )
        writer.writeheader()
        writer.writerows(audited_rows)

    with open(args.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Training rows checked: {training_rows}")
    print(f"Auditable rows: {auditable_rows}")
    print(f"Unresolved rows: {unresolved_rows}")
    print(f"Mismatch rows: {mismatch_rows}")
    print(f"Mean mistake: {mean_mistake:.6f}")
    print(f"Mean mistake (all labeled denominator): {mean_mistake_all_labeled:.6f}")
    print(f"Audit CSV: {args.output_csv}")
    print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
