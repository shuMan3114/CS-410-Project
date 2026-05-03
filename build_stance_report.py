#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import textwrap
from collections import Counter
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output_csvs"
DOC_DIR = ROOT / "doc"
OUTPUT_PDF = DOC_DIR / "CS_410_Stance_Pipeline_Report.pdf"

PAGE_W = 595
PAGE_H = 842
MARGIN_X = 54
TOP_Y = 60
BOTTOM_Y = 785


def count_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1


def count_rows_if_exists(path: Path) -> int | None:
    return count_rows(path) if path.exists() else None


def counter_for(path: Path, field: str) -> Counter:
    with path.open(newline="", encoding="utf-8") as handle:
        return Counter(row[field] for row in csv.DictReader(handle))


def distribution_rows(path: Path, field: str) -> list[list[str]]:
    ctr = counter_for(path, field)
    total = sum(ctr.values())
    rows = []
    for label in ["supportive", "opposing", "neutral", "mixed", "unclear"]:
        count = ctr.get(label, 0)
        pct = f"{(100.0 * count / total):.1f}%" if total else "0.0%"
        rows.append([label, str(count), pct])
    return rows


def new_page(doc: fitz.Document, title: str | None = None, page_no: int | None = None) -> fitz.Page:
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.draw_line((MARGIN_X, 42), (PAGE_W - MARGIN_X, 42), color=(0.4, 0.4, 0.4), width=0.8)
    if title:
        page.insert_text((MARGIN_X, 30), title, fontsize=10, fontname="Times-Italic", color=(0.25, 0.25, 0.25))
    if page_no is not None:
        page.insert_text((PAGE_W - MARGIN_X - 10, PAGE_H - 24), str(page_no), fontsize=10, fontname="Times-Roman")
    return page


def add_heading(page: fitz.Page, y: float, text: str, size: float = 16) -> float:
    page.insert_text((MARGIN_X, y), text, fontsize=size, fontname="Times-Bold")
    return y + size + 8


def add_subheading(page: fitz.Page, y: float, text: str, size: float = 12) -> float:
    page.insert_text((MARGIN_X, y), text, fontsize=size, fontname="Times-Bold")
    return y + size + 6


def add_paragraph(page: fitz.Page, y: float, text: str, width_chars: int = 92, size: float = 11) -> float:
    lines = textwrap.wrap(text, width=width_chars)
    for line in lines:
        page.insert_text((MARGIN_X, y), line, fontsize=size, fontname="Times-Roman")
        y += size + 3
    return y + 4


def add_bullets(page: fitz.Page, y: float, bullets: list[str], width_chars: int = 88, size: float = 11) -> float:
    for bullet in bullets:
        wrapped = textwrap.wrap(bullet, width=width_chars)
        if not wrapped:
            y += size + 4
            continue
        page.insert_text((MARGIN_X, y), f"{wrapped[0]}", fontsize=size, fontname="Times-Roman")
        y += size + 3
        for line in wrapped[1:]:
            page.insert_text((MARGIN_X + 14, y), line, fontsize=size, fontname="Times-Roman")
            y += size + 3
        y += 4
    return y


def draw_table(page: fitz.Page, x: float, y: float, headers: list[str], rows: list[list[str]], col_widths: list[float], row_h: float = 22) -> float:
    total_w = sum(col_widths)
    total_h = row_h * (len(rows) + 1)
    outer = fitz.Rect(x, y, x + total_w, y + total_h)
    page.draw_rect(outer, color=(0.3, 0.3, 0.3), width=0.8)
    page.draw_rect(fitz.Rect(x, y, x + total_w, y + row_h), color=(0.3, 0.3, 0.3), fill=(0.92, 0.92, 0.92), width=0.8)

    label_width = max(len(r[0]) for r in ([headers] + rows)) + 2
    count_width = max(len(r[1]) for r in ([headers] + rows)) + 2
    pct_width = max(len(r[2]) for r in ([headers] + rows)) + 2

    def fmt(row: list[str]) -> str:
        return f"{row[0]:<{label_width}} {row[1]:>{count_width}} {row[2]:>{pct_width}}"

    current_y = y + 14
    page.insert_text((x + 10, current_y), fmt(headers), fontsize=10.5, fontname="Courier-Bold", color=(0, 0, 0))

    current_y = y + row_h
    for idx, row in enumerate(rows):
        page.draw_line((x, current_y), (x + total_w, current_y), color=(0.8, 0.8, 0.8), width=0.5)
        page.insert_text((x + 10, current_y + 14), fmt(row), fontsize=10.5, fontname="Courier", color=(0, 0, 0))
        current_y += row_h
    return y + total_h


def main() -> None:
    DOC_DIR.mkdir(exist_ok=True)

    papers_path = OUT_DIR / "papers.csv"
    phase1_path = OUT_DIR / "stance_phase1_bits.csv"
    phase2_path = OUT_DIR / "stance_phase2_counts.csv"
    phase3_12_path = OUT_DIR / "stance_phase3_window_12.csv"
    eval_path = OUT_DIR / "stance_labels_eval_sample.csv"

    total_papers = count_rows_if_exists(papers_path)
    phase_rows = count_rows(phase1_path)
    eval_rows = count_rows(eval_path)

    phase1_rows = distribution_rows(phase1_path, "Phase1StanceGuess")
    phase2_rows = distribution_rows(phase2_path, "Phase2StanceGuess")
    phase3_rows = distribution_rows(phase3_12_path, "WeightedStanceGuess")
    eval_sampling_rows = distribution_rows(eval_path, "sampling_label")
    eval_phase1_rows = distribution_rows(eval_path, "phase_1_label")
    eval_phase2_rows = distribution_rows(eval_path, "phase_2_label")
    eval_phase3_rows = distribution_rows(eval_path, "phase_3_12_label")

    today = dt.datetime.now().strftime("%B %d, %Y")
    doc = fitz.open()

    page = new_page(doc, "ProtEvidenceDB Stance Pipeline", 1)
    y = add_heading(page, TOP_Y, "PT1 Stance Modeling Update", 18)
    y = add_heading(page, y, "ProtEvidenceDB: Abstract-Based Stance Pipeline Report", 15)
    y = add_paragraph(page, y + 8, f"Generated: {today}", width_chars=110, size=11)
    y += 8
    intro = (
        "This report summarizes the current abstract-level stance pipeline over cancer biomarker literature. "
        "The final setup no longer trusts the retrieval-time paper-to-pair link by itself. Instead, it validates "
        "gene and drug mentions directly inside each abstract, matches those mentions against the existing "
        "drug-gene interaction database, and only then computes the stance features and labels."
    )
    y = add_paragraph(page, y, intro)
    y = add_subheading(page, y + 8, "1. Intuition")
    intuition = [
        "1. Retrieval links are noisy. A paper found by a query does not always explicitly discuss the linked gene-drug pair.",
        "2. Stance should be tied to local evidence, not the whole abstract. The pipeline therefore uses sentence-bounded matches and token windows.",
        "3. Nearby phrases should count more than distant ones. Phase 3 uses a distance-weighted score: weight = base / (1 + distance_to_gene + distance_to_drug).",
        "4. Generic neutral cues are downweighted so words like evaluated or review do not dominate the signal.",
        "5. Evaluation is separated from raw extraction. Full phase outputs are written for all eligible papers; a separate 2,000-row CSV is produced for LLM/manual review.",
    ]
    add_bullets(page, y, intuition)

    page = new_page(doc, "ProtEvidenceDB Stance Pipeline", 2)
    y = add_heading(page, TOP_Y, "2. Phase Definitions", 16)
    phase_text = [
        "Phase 1: Sentence-Level Bits. A paper receives binary indicators for whether a validated gene-drug pair appears in the same sentence as a positive, negative, or neutral phrase.",
        "Phase 2: Sentence-Level Counts. This phase counts how many times those sentence-level co-occurrences appear. It is less strict than Phase 3 and is currently used as the sampling label for evaluation balancing.",
        "Phase 3: Sentence-Bounded Token Windows. For each window size in {2, 5, 8, 10, 12, 16, 20}, the pipeline scores stance phrases by how close they are to both the gene and drug mention. Same-clause and between-entity phrase placements receive a modest boost.",
        "Clarity / Unclear Logic. A Phase 3 label is marked as unclear when the strongest stance score is too weak or when the top two stance scores are too close to separate confidently.",
    ]
    for block in phase_text:
        y = add_paragraph(page, y, block)
        y += 4
    y = add_heading(page, y + 8, "3. Dataset Sizes", 16)
    size_bullets = []
    if total_papers is not None:
        size_bullets.append(f"PubMed papers.csv rows: {total_papers}")
    else:
        size_bullets.append("PubMed papers.csv rows: unavailable in current workspace snapshot")
    size_bullets.extend([
        f"Papers with abstract-mentioned, interaction-validated gene-drug pairs: {phase_rows}",
        f"Evaluation sample size: {eval_rows}",
        "The abstract-validated matching step removes many papers where the retrieval query found a paper, but the abstract itself does not state the relevant gene-drug pair.",
    ])
    add_bullets(page, y, size_bullets)

    page = new_page(doc, "ProtEvidenceDB Stance Pipeline", 3)
    y = add_heading(page, TOP_Y, "4. Full-Dataset Distributions", 16)
    y = add_subheading(page, y + 4, "Phase 1")
    y = draw_table(page, MARGIN_X, y, ["Label", "Count", "Percent"], phase1_rows, [210, 110, 110]) + 20
    y = add_subheading(page, y, "Phase 2")
    y = draw_table(page, MARGIN_X, y, ["Label", "Count", "Percent"], phase2_rows, [210, 110, 110]) + 20
    y = add_subheading(page, y, "Phase 3 (Window = 12)")
    draw_table(page, MARGIN_X, y, ["Label", "Count", "Percent"], phase3_rows, [210, 110, 110])

    page = new_page(doc, "ProtEvidenceDB Stance Pipeline", 4)
    y = add_heading(page, TOP_Y, "5. Evaluation Sample Distributions", 16)
    y = add_subheading(page, y + 4, "Sampling Label (balanced by Phase 2)")
    y = draw_table(page, MARGIN_X, y, ["Label", "Count", "Percent"], eval_sampling_rows, [210, 110, 110]) + 20
    y = add_subheading(page, y, "Evaluation Labels: Phase 1")
    y = draw_table(page, MARGIN_X, y, ["Label", "Count", "Percent"], eval_phase1_rows, [210, 110, 110]) + 20
    y = add_subheading(page, y, "Evaluation Labels: Phase 2")
    draw_table(page, MARGIN_X, y, ["Label", "Count", "Percent"], eval_phase2_rows, [210, 110, 110])

    page = new_page(doc, "ProtEvidenceDB Stance Pipeline", 5)
    y = add_heading(page, TOP_Y, "6. Evaluation Labels: Phase 3 (Window = 12)", 16)
    y = draw_table(page, MARGIN_X, y + 8, ["Label", "Count", "Percent"], eval_phase3_rows, [210, 110, 110]) + 28
    y = add_heading(page, y, "7. Interpretation", 16)
    observations = [
        "1. Phase 1 and Phase 2 show materially more non-unclear labels after switching to abstract-validated gene-drug pairs.",
        "2. Phase 3 remains much more conservative because it requires local proximity and confidence separation.",
        "3. The balanced evaluation sample is driven by Phase 2 because Phase 3 alone does not yet produce enough non-unclear cases for a useful manual review set.",
        "4. This setup is suitable for the next stage: LLM or manual adjudication over the 2,000-row evaluation sheet, followed by supervised classifier training.",
    ]
    add_bullets(page, y, observations)

    doc.save(OUTPUT_PDF)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
