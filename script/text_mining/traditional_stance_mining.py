#!/usr/bin/env python3
"""
Traditional stance text-mining pipeline for biomarker-drug abstracts.

Pipeline stages:
1) POS-aware preprocessing (stopword removal + POS tagging + lemmatization).
2) Syntagmatic association mining with PMI (from the course notes).
3) BM25-style weighted stance scoring per paper.

Outputs:
  - traditional_preprocessed_sentences.csv
  - traditional_stance_patterns.csv
  - traditional_stance_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import Counter, defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-+/#]*")
CONTENT_POS_PREFIXES = ("JJ", "RB", "VB", "NN")

SEED_TERMS = {
    "supportive": [
        "effective",
        "improved",
        "improves",
        "improvement",
        "benefit",
        "beneficial",
        "enhanced",
        "response",
        "responded",
        "sensitive",
        "sensitivity",
        "inhibition",
        "suppressed",
        "reduced",
        "decreased",
        "synergistic",
        "promising",
        "favorable",
    ],
    "opposing": [
        "resistant",
        "resistance",
        "ineffective",
        "no benefit",
        "failed",
        "failure",
        "worse",
        "adverse",
        "toxic",
        "limited",
        "poor response",
        "progression",
        "did not improve",
        "not effective",
        "contraindicated",
        "unsuccessful",
    ],
    "neutral": [
        "associated",
        "association",
        "correlated",
        "investigated",
        "evaluated",
        "explored",
        "review",
        "objective",
        "background",
        "inconclusive",
        "unclear",
        "uncertain",
        "mixed results",
        "preliminary",
        "suggests",
        "may be associated",
    ],
}


def ensure_nltk_resources() -> None:
    nltk.download("stopwords", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(normalized) if s.strip()]


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def to_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return "a"
    if treebank_tag.startswith("V"):
        return "v"
    if treebank_tag.startswith("N"):
        return "n"
    if treebank_tag.startswith("R"):
        return "r"
    return "n"


def contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.search(rf"\b{escaped}\b", text, flags=re.IGNORECASE) is not None


def all_phrase_hits(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if contains_phrase(text, term)]


def term_token_positions(text: str, terms: list[str], token_spans: list[tuple[str, int, int]]) -> list[int]:
    positions = []
    for term in terms:
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        for match in re.finditer(rf"\b{escaped}\b", text, flags=re.IGNORECASE):
            start = match.start()
            end = match.end()
            for idx, (_, token_start, token_end) in enumerate(token_spans):
                if token_start <= start < token_end or start <= token_start < end:
                    positions.append(idx)
                    break
    return sorted(set(positions))


def require_columns(fieldnames: list[str], required: list[str], csv_path: str) -> None:
    missing = [col for col in required if col not in fieldnames]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")


def load_papers(path: str) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames or [], ["PaperID", "DOI", "Abstract"], path)
        papers = {}
        for row in reader:
            papers[row["PaperID"]] = row
    return papers


def load_links(path: str) -> dict[str, dict[str, list[str]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames or [], ["PaperID", "GeneSymbol", "DrugName"], path)
        links = defaultdict(lambda: {"genes": [], "drugs": []})
        seen = defaultdict(lambda: {"genes": set(), "drugs": set()})
        for row in reader:
            paper_id = row["PaperID"]
            gene = row["GeneSymbol"].strip()
            drug = row["DrugName"].strip()
            if gene and gene not in seen[paper_id]["genes"]:
                seen[paper_id]["genes"].add(gene)
                links[paper_id]["genes"].append(gene)
            if drug and drug not in seen[paper_id]["drugs"]:
                seen[paper_id]["drugs"].add(drug)
                links[paper_id]["drugs"].append(drug)
    return links


def tag_and_filter_tokens(
    tokens: list[str],
    stop_words: set[str],
    lemmatizer: WordNetLemmatizer,
    allowed_indices: set[int],
) -> tuple[list[str], list[str]]:
    tagged = nltk.pos_tag(tokens)
    filtered = []
    tagged_str = []
    for idx, (token, pos_tag) in enumerate(tagged):
        tagged_str.append(f"{token}/{pos_tag}")
        if idx not in allowed_indices:
            continue
        if token in stop_words:
            continue
        if len(token) <= 1:
            continue
        if not any(ch.isalnum() for ch in token):
            continue
        lemma = lemmatizer.lemmatize(token, to_wordnet_pos(pos_tag))
        if not pos_tag.startswith(CONTENT_POS_PREFIXES):
            continue
        filtered.append(lemma)
    return filtered, tagged_str


def softmax_scores(class_scores: dict[str, float]) -> dict[str, float]:
    max_score = max(class_scores.values())
    exps = {label: math.exp(score - max_score) for label, score in class_scores.items()}
    total = sum(exps.values())
    return {label: value / total for label, value in exps.items()}


def infer_label_from_probabilities(
    class_probs: dict[str, float],
    min_signal: float,
    min_margin: float,
) -> tuple[str, float, float]:
    ordered = sorted(class_probs.items(), key=lambda pair: pair[1], reverse=True)
    top = ordered[0][1]
    second = ordered[1][1]
    margin = top - second
    if top < min_signal or margin < min_margin:
        return "unclear", top, margin
    return ordered[0][0], top, margin


def write_csv(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TraditionalStanceMiner:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def Run(self) -> None:
        args = self.args

        ensure_nltk_resources()
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        papers = load_papers(args.papers)
        links = load_links(args.paper_sources)
        eligible_ids = [
            pid for pid in papers
            if normalize_text(papers[pid]["Abstract"]) and links[pid]["genes"] and links[pid]["drugs"]
        ]
        if args.debug_max_papers > 0:
            eligible_ids = eligible_ids[: args.debug_max_papers]

        print(f"Eligible papers: {len(eligible_ids)}")

        segment_rows: list[dict[str, str]] = []
        paper_segment_tokens: dict[str, list[str]] = defaultdict(list)
        paper_segment_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        paper_evidence_sentences: dict[str, list[str]] = defaultdict(list)
        paper_seed_hit_counts: dict[str, Counter[str]] = defaultdict(Counter)
        paper_token_presence: dict[str, set[str]] = defaultdict(set)

        segment_count = 0
        class_segment_count = Counter()
        token_segment_count = Counter()
        token_class_count = {label: Counter() for label in SEED_TERMS}

        for idx, paper_id in enumerate(eligible_ids, start=1):
            paper = papers[paper_id]
            genes = links[paper_id]["genes"]
            drugs = links[paper_id]["drugs"]
            sentences = split_sentences(paper["Abstract"])
            if args.debug_max_sentences_per_paper > 0:
                sentences = sentences[: args.debug_max_sentences_per_paper]

            for sent_idx, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                gene_hits = [g for g in genes if contains_phrase(sentence_lower, g.lower())]
                drug_hits = [d for d in drugs if contains_phrase(sentence_lower, d.lower())]
                if not gene_hits or not drug_hits:
                    continue

                tokens_with_spans = tokenize_with_spans(sentence_lower)
                raw_tokens = [token for token, _, _ in tokens_with_spans]
                if not raw_tokens:
                    continue

                gene_positions = term_token_positions(sentence_lower, gene_hits, tokens_with_spans)
                drug_positions = term_token_positions(sentence_lower, drug_hits, tokens_with_spans)
                if not gene_positions or not drug_positions:
                    continue

                allowed_indices = set()
                for token_idx in range(len(raw_tokens)):
                    nearest_gene = min(abs(token_idx - gp) for gp in gene_positions)
                    nearest_drug = min(abs(token_idx - dp) for dp in drug_positions)
                    if nearest_gene <= args.window_size and nearest_drug <= args.window_size:
                        allowed_indices.add(token_idx)

                filtered_tokens, tagged_tokens = tag_and_filter_tokens(
                    raw_tokens,
                    stop_words,
                    lemmatizer,
                    allowed_indices,
                )
                if not filtered_tokens:
                    continue

                seed_hits = {
                    "supportive": all_phrase_hits(sentence_lower, SEED_TERMS["supportive"]),
                    "opposing": all_phrase_hits(sentence_lower, SEED_TERMS["opposing"]),
                    "neutral": all_phrase_hits(sentence_lower, SEED_TERMS["neutral"]),
                }

                segment_count += 1
                unique_segment_tokens = set(filtered_tokens)
                for token in unique_segment_tokens:
                    token_segment_count[token] += 1
                    paper_token_presence[paper_id].add(token)

                for label in ("supportive", "opposing", "neutral"):
                    if seed_hits[label]:
                        class_segment_count[label] += 1
                        for token in unique_segment_tokens:
                            token_class_count[label][token] += 1
                        paper_seed_hit_counts[paper_id][label] += len(seed_hits[label])
                        if len(paper_evidence_sentences[paper_id]) < 6:
                            paper_evidence_sentences[paper_id].append(sentence)

                paper_segment_tokens[paper_id].extend(filtered_tokens)
                paper_segment_token_counts[paper_id].update(filtered_tokens)

                segment_rows.append({
                    "PaperID": paper_id,
                    "SentenceID": str(sent_idx),
                    "Sentence": sentence,
                    "GeneHits": "|".join(gene_hits),
                    "DrugHits": "|".join(drug_hits),
                    "TokenPOS": " ".join(tagged_tokens),
                    "FilteredTokens": " ".join(filtered_tokens),
                    "SupportiveSeedHits": "|".join(seed_hits["supportive"]),
                    "OpposingSeedHits": "|".join(seed_hits["opposing"]),
                    "NeutralSeedHits": "|".join(seed_hits["neutral"]),
                })

            if args.debug_print_every > 0 and idx % args.debug_print_every == 0:
                print(f"Processed papers: {idx}/{len(eligible_ids)}")

        if segment_count == 0:
            raise ValueError("No gene-drug co-mentioned segments were found in eligible papers.")

        paper_context_count = len([pid for pid in eligible_ids if paper_token_presence[pid]])
        token_idf = {}
        for token, df_count in Counter(token for pid in paper_token_presence for token in paper_token_presence[pid]).items():
            token_idf[token] = math.log((paper_context_count + 1.0) / (df_count + 1.0)) + 1.0

        pmi_scores = {label: {} for label in SEED_TERMS}
        selected_patterns = {label: [] for label in SEED_TERMS}
        alpha = args.alpha

        for label in ("supportive", "opposing", "neutral"):
            for token, count_w in token_segment_count.items():
                if count_w < args.min_token_segment_count:
                    continue
                count_c = class_segment_count[label]
                count_wc = token_class_count[label][token]
                if count_wc < args.min_token_class_count:
                    continue

                p_w = (count_w + alpha) / (segment_count + 2.0 * alpha)
                p_c = (count_c + alpha) / (segment_count + 2.0 * alpha)
                p_wc = (count_wc + alpha) / (segment_count + 4.0 * alpha)
                pmi = math.log2(p_wc / (p_w * p_c))
                pmi_scores[label][token] = pmi

            ranked_tokens = sorted(
                pmi_scores[label].items(),
                key=lambda item: item[1],
                reverse=True,
            )
            for token, pmi in ranked_tokens:
                if pmi < args.min_positive_pmi:
                    continue
                selected_patterns[label].append(token)
                if len(selected_patterns[label]) >= args.top_k_patterns:
                    break

        label_rows = []
        for paper_id in eligible_ids:
            if not paper_segment_tokens[paper_id]:
                continue
            paper = papers[paper_id]
            genes = links[paper_id]["genes"]
            drugs = links[paper_id]["drugs"]
            token_counts = paper_segment_token_counts[paper_id]

            class_scores = {}
            for label in ("supportive", "opposing", "neutral"):
                score = 0.0
                for token in selected_patterns[label]:
                    tf_raw = token_counts[token]
                    if tf_raw <= 0:
                        continue
                    tf_bm25 = ((args.bm25_k + 1.0) * tf_raw) / (tf_raw + args.bm25_k)
                    score += tf_bm25 * token_idf[token] * max(0.0, pmi_scores[label][token])
                score += 0.35 * paper_seed_hit_counts[paper_id][label]
                class_scores[label] = score

            class_probs = softmax_scores(class_scores)
            prediction, confidence_score, confidence_margin = infer_label_from_probabilities(
                class_probs,
                args.min_signal,
                args.min_margin,
            )

            all_tokens = set(token_counts.keys())
            evidence_tokens = []
            for label in ("supportive", "opposing", "neutral"):
                for token in selected_patterns[label][:35]:
                    if token in all_tokens:
                        evidence_tokens.append(f"{label}:{token}")

            label_rows.append({
                "PaperID": paper_id,
                "DOI": paper["DOI"],
                "Biomarkers": "|".join(genes),
                "Drugs": "|".join(drugs),
                "SupportiveScore": f"{class_scores['supportive']:.6f}",
                "OpposingScore": f"{class_scores['opposing']:.6f}",
                "NeutralScore": f"{class_scores['neutral']:.6f}",
                "P_supportive": f"{class_probs['supportive']:.6f}",
                "P_opposing": f"{class_probs['opposing']:.6f}",
                "P_neutral": f"{class_probs['neutral']:.6f}",
                "PredictedStance": prediction,
                "ConfidenceScore": f"{confidence_score:.6f}",
                "ConfidenceMargin": f"{confidence_margin:.6f}",
                "EvidenceTokens": " | ".join(evidence_tokens[:20]),
                "EvidenceSentences": " || ".join(paper_evidence_sentences[paper_id][:4]),
                "LLMReviewLabel": "",
                "ManualNotes": "",
            })

        pattern_rows = []
        for label in ("supportive", "opposing", "neutral"):
            for token in selected_patterns[label]:
                pattern_rows.append({
                    "Label": label,
                    "Token": token,
                    "PMI": f"{pmi_scores[label][token]:.6f}",
                    "TokenSegmentCount": str(token_segment_count[token]),
                    "LabelSegmentCount": str(class_segment_count[label]),
                    "TokenLabelCount": str(token_class_count[label][token]),
                    "IDF": f"{token_idf[token]:.6f}",
                })

        os.makedirs(args.output_dir, exist_ok=True)
        preprocessed_path = os.path.join(args.output_dir, "traditional_preprocessed_sentences.csv")
        patterns_path = os.path.join(args.output_dir, "traditional_stance_patterns.csv")
        labels_path = os.path.join(args.output_dir, "traditional_stance_labels.csv")

        write_csv(
            preprocessed_path,
            segment_rows,
            [
                "PaperID",
                "SentenceID",
                "Sentence",
                "GeneHits",
                "DrugHits",
                "TokenPOS",
                "FilteredTokens",
                "SupportiveSeedHits",
                "OpposingSeedHits",
                "NeutralSeedHits",
            ],
        )

        write_csv(
            patterns_path,
            pattern_rows,
            [
                "Label",
                "Token",
                "PMI",
                "TokenSegmentCount",
                "LabelSegmentCount",
                "TokenLabelCount",
                "IDF",
            ],
        )

        write_csv(
            labels_path,
            label_rows,
            [
                "PaperID",
                "DOI",
                "Biomarkers",
                "Drugs",
                "SupportiveScore",
                "OpposingScore",
                "NeutralScore",
                "P_supportive",
                "P_opposing",
                "P_neutral",
                "PredictedStance",
                "ConfidenceScore",
                "ConfidenceMargin",
                "EvidenceTokens",
                "EvidenceSentences",
                "LLMReviewLabel",
                "ManualNotes",
            ],
        )

        print(f"Segments mined: {segment_count}")
        print(f"Class segment counts: {dict(class_segment_count)}")
        print(f"Preprocessed segments: {preprocessed_path}")
        print(f"Traditional patterns: {patterns_path}")
        print(f"Traditional labels: {labels_path}")
        for label in ("supportive", "opposing", "neutral"):
            preview = selected_patterns[label][:12]
            print(f"Top patterns [{label}]: {preview}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traditional stance mining with POS preprocessing + PMI/BM25")
    parser.add_argument("--papers", default="data/raw/papers.csv")
    parser.add_argument("--paper-sources", default="data/raw/paper_sources.csv")
    parser.add_argument("--output-dir", default="output_csvs")
    parser.add_argument("--debug-max-papers", type=int, default=0, help="0 means all eligible papers")
    parser.add_argument("--debug-max-sentences-per-paper", type=int, default=0, help="0 means all sentences")
    parser.add_argument("--debug-print-every", type=int, default=250)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.25, help="smoothing pseudo-count for PMI")
    parser.add_argument("--min-token-segment-count", type=int, default=5)
    parser.add_argument("--min-token-class-count", type=int, default=3)
    parser.add_argument("--min-positive-pmi", type=float, default=0.05)
    parser.add_argument("--top-k-patterns", type=int, default=120)
    parser.add_argument("--bm25-k", type=float, default=1.2)
    parser.add_argument("--min-signal", type=float, default=0.60, help="min top class probability")
    parser.add_argument("--min-margin", type=float, default=0.20, help="min probability gap from runner-up")
    return parser


if __name__ == "__main__":
    miner = TraditionalStanceMiner(build_arg_parser().parse_args())
    miner.Run()
