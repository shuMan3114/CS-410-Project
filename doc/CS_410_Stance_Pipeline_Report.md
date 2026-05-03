# CS-410 Stance Modeling Pipeline Report

Generated: 2026-05-03 10:12

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

- Total traditional rows: **10455**
- Traditional label distribution:
  - supportive: 2068 (19.78%)
  - opposing: 1233 (11.79%)
  - neutral: 1348 (12.89%)
  - unclear: 5806 (55.53%)

## 3) Traditional Label Audit (All Training-Eligible Labels)

Audit source: `traditional_label_audit_summary.json` from `audit_traditional_labels.py`.

- Training-eligible labeled rows checked: **4649**
- Auditable rows (clear rubric decision): **2811**
- Unresolved rows (insufficient/conflicting evidence cues): **1838**
- Mismatch rows: **488**
- **Mean mistake (auditable denominator)**: **0.173604** (17.36%)
- **Mean mistake (all-labeled denominator)**: **0.104969** (10.50%)

Interpretation for reporting:
- If you want a strict “checked-only” error rate, use **17.36%**.
- If you want a conservative full-set rate, use **10.50%**.

## 4) Downstream Classifier Performance

From `bow_stance_eval.json`:

- Train accuracy: **0.694273** on 3719 rows
- Validation accuracy: **0.511828** on 930 rows

## 5) Final Paper Stances (After Unclear Backfill)

- Total final rows: **49614**
- Final source composition:
  - `traditional_non_unclear`: 4649 (9.37%)
  - `classifier_backfill_unclear`: 5806 (11.70%)
  - `classifier_only`: 39159 (78.93%)

- Final label distribution:
  - supportive: 17295 (34.86%)
  - opposing: 14916 (30.06%)
  - neutral: 17403 (35.08%)

## 6) Key Output Files

- `output_csvs/traditional_stance_labels.csv`
- `output_csvs/traditional_label_audit.csv`
- `output_csvs/traditional_label_audit_summary.json`
- `output_csvs/bow_stance_predictions.csv`
- `output_csvs/bow_stance_eval.json`
- `output_csvs/final_paper_stances.csv`
