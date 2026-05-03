# CS-410 Stance Extraction Pipeline

This document describes the full stance extraction workflow in `script/text_mining/`.

The recommender is now treated as a separate downstream system and consumes only:

- `output_csvs/final_paper_stances.csv`

---

## 1) Inputs

Primary raw inputs:

- `data/raw/papers.csv`
- `data/raw/paper_sources.csv`
- `data/raw/biomarkers.csv`
- `data/raw/drug_gene_interactions.csv`

Core intermediate/derived outputs:

- `output_csvs/traditional_preprocessed_sentences.csv`
- `output_csvs/traditional_stance_patterns.csv`
- `output_csvs/traditional_stance_labels.csv`
- `output_csvs/bow_stance_predictions.csv`
- `output_csvs/final_paper_stances.csv`

---

## 2) Pipeline Stages

### Stage A: Traditional stance mining

Script:

- `script/text_mining/traditional_stance_mining.py`

What it does:

- Extracts gene-drug co-mentioned sentence segments.
- Applies POS-aware filtering and lemmatization.
- Mines stance lexicon patterns with PMI.
- Produces paper-level stance probabilities and confidence.

### Stage B: Weak-label quality audit

Script:

- `script/text_mining/audit_traditional_labels.py`

What it does:

- Audits training-eligible traditional labels using evidence-sentence rubric.
- Reports mismatch rates for quality tracking.

### Stage C: Downstream BoW classifier

Script:

- `script/text_mining/train_bow_stance_classifier.py`

What it does:

- Trains multinomial Naive Bayes on weak labels (non-unclear, confidence-filtered).
- Predicts stance probabilities for all papers.

### Stage D: Final stance consolidation

Script:

- `script/text_mining/build_final_paper_stances.py`

What it does:

- Keeps traditional non-unclear labels.
- Backfills traditional unclear rows with downstream classifier predictions.
- Emits final paper-level stance file for recommender consumption.

---

## 3) Current Metrics (Latest Run)

### Traditional mining

- Segments mined: **16332**
- Segment class counts:
  - supportive: **3964**
  - opposing: **1841**
  - neutral: **2565**

### Traditional paper-level labels

- Total rows: **10455**
- Label distribution:
  - supportive: **2068** (19.78%)
  - opposing: **1233** (11.79%)
  - neutral: **1348** (12.89%)
  - unclear: **5806** (55.53%)

### Traditional label audit

- Training rows checked: **4649**
- Auditable rows: **2811**
- Mismatch rows: **488**
- Mean mistake (auditable denominator): **17.36%**
- Mean mistake (all-labeled denominator): **10.50%**

### BoW classifier

- Train accuracy: **0.6943**
- Validation accuracy: **0.5118**

### Final stance file

- Total papers: **49614**
- Final label distribution:
  - supportive: **17295** (34.86%)
  - opposing: **14916** (30.06%)
  - neutral: **17403** (35.08%)
- Final source composition:
  - traditional_non_unclear: **4649**
  - classifier_backfill_unclear: **5806**
  - classifier_only: **39159**

---

## 4) Repro Commands

```bash
.venv/bin/python "script/text_mining/traditional_stance_mining.py" --output-dir output_csvs
.venv/bin/python "script/text_mining/audit_traditional_labels.py"
.venv/bin/python "script/text_mining/train_bow_stance_classifier.py" --drop-unclear
.venv/bin/python "script/text_mining/build_final_paper_stances.py"
```

---

## 5) Separation of Responsibilities

- `script/text_mining/*` is the stance extraction/training pipeline.
- `script/recommendation/*` is the recommendation pipeline.
- Recommender stance dependency is only:
  - `output_csvs/final_paper_stances.csv`
