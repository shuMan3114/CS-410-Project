#!/usr/bin/env python3
"""
Train a bag-of-words Multinomial Naive Bayes stance classifier.

The code is organized in a class-first workflow (similar to HW1 scripts):
initialize resources in __init__, then run staged methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

import nltk
from nltk.corpus import stopwords


class BoWStanceClassifier:
    TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-+/#]*")
    SUPPORTED_LABELS = {"supportive", "opposing", "neutral"}

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stop_words: set[str] = set()
        self.papers: dict[str, dict[str, str]] = {}
        self.links: dict[str, dict[str, list[str]]] = {}
        self.weak_labels: dict[str, tuple[str, float]] = {}

    @staticmethod
    def RequireColumns(fieldnames: list[str], required: list[str], csv_path: str) -> None:
        missing = [col for col in required if col not in fieldnames]
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {missing}")

    @staticmethod
    def NormalizeText(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    @classmethod
    def Tokenize(cls, text: str) -> list[str]:
        return [m.group(0).lower() for m in cls.TOKEN_RE.finditer(text)]

    def EnsureResources(self) -> None:
        nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words("english"))

    def LoadPapers(self) -> None:
        with open(self.args.papers, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.RequireColumns(reader.fieldnames or [], ["PaperID", "DOI", "Title", "Abstract"], self.args.papers)
            papers: dict[str, dict[str, str]] = {}
            for row in reader:
                papers[row["PaperID"]] = row
            self.papers = papers

    def LoadLinks(self) -> None:
        with open(self.args.paper_sources, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.RequireColumns(
                reader.fieldnames or [],
                ["PaperID", "GeneSymbol", "DrugName"],
                self.args.paper_sources,
            )
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
            self.links = links

    def LoadWeakLabels(self) -> None:
        with open(self.args.weak_labels, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.RequireColumns(
                reader.fieldnames or [],
                [self.args.labels_paperid_col, self.args.labels_label_col, self.args.labels_confidence_col],
                self.args.weak_labels,
            )
            labels: dict[str, tuple[str, float]] = {}
            for row in reader:
                label = row[self.args.labels_label_col].strip().lower()
                confidence = float(row[self.args.labels_confidence_col])
                if not (0.0 <= confidence <= 1.0):
                    raise ValueError(
                        f"{self.args.weak_labels} has out-of-range confidence for "
                        f"PaperID={row[self.args.labels_paperid_col]}: {confidence}. Expected [0,1]."
                    )
                if self.args.drop_unclear and label == "unclear":
                    continue
                if label not in self.SUPPORTED_LABELS:
                    continue
                if confidence < self.args.min_confidence:
                    continue
                labels[row[self.args.labels_paperid_col]] = (label, confidence)
            self.weak_labels = labels

    def FilterTokens(self, tokens: list[str]) -> list[str]:
        filtered = []
        for token in tokens:
            if token in self.stop_words:
                continue
            if len(token) <= 1:
                continue
            if not self.args.keep_numeric and token.isdigit():
                continue
            filtered.append(token)
        return filtered

    def BuildTrainingExamples(self) -> list[dict[str, object]]:
        if len(self.weak_labels) < 20:
            raise ValueError(f"Need at least 20 labeled papers to train; got {len(self.weak_labels)}")

        candidate_ids = [paper_id for paper_id in self.weak_labels if paper_id in self.papers]
        candidate_ids.sort()
        if self.args.debug_max_papers > 0:
            candidate_ids = candidate_ids[: self.args.debug_max_papers]

        examples = []
        for paper_id in candidate_ids:
            abstract = self.NormalizeText(self.papers[paper_id]["Abstract"])
            if not abstract:
                continue
            tokens = self.FilterTokens(self.Tokenize(abstract))
            if not tokens:
                continue
            label, confidence = self.weak_labels[paper_id]
            examples.append(
                {
                    "paper_id": paper_id,
                    "label": label,
                    "confidence": confidence,
                    "tokens": tokens,
                }
            )
        if len(examples) < 20:
            raise ValueError(f"Need at least 20 usable examples after preprocessing; got {len(examples)}")
        return examples

    def SplitTrainValidation(
        self,
        examples: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        rng = random.Random(self.args.seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1.0 - self.args.validation_fraction))
        split_idx = max(1, min(split_idx, len(shuffled) - 1))
        return shuffled[:split_idx], shuffled[split_idx:]

    def BuildVocab(self, documents: list[list[str]]) -> list[str]:
        doc_freq = Counter()
        for doc in documents:
            doc_freq.update(set(doc))
        kept = [token for token, count in doc_freq.items() if count >= self.args.min_df]
        kept.sort(key=lambda token: (-doc_freq[token], token))
        if self.args.max_vocab > 0:
            kept = kept[: self.args.max_vocab]
        return kept

    def TrainMultinomialNB(
        self,
        examples: list[dict[str, object]],
        classes: list[str],
        vocab: list[str],
    ) -> dict[str, object]:
        vocab_set = set(vocab)
        token_counts = {label: Counter() for label in classes}
        total_tokens = Counter()
        class_doc_counts = Counter()

        for ex in examples:
            label = ex["label"]
            tokens = [tok for tok in ex["tokens"] if tok in vocab_set]
            class_doc_counts[label] += 1
            token_counts[label].update(tokens)
            total_tokens[label] += len(tokens)

        total_docs = len(examples)
        class_count = len(classes)
        vocab_size = len(vocab)
        log_priors = {}
        log_likelihood = {label: {} for label in classes}
        unknown_log_likelihood = {}

        for label in classes:
            prior = (class_doc_counts[label] + self.args.alpha) / (total_docs + self.args.alpha * class_count)
            log_priors[label] = math.log(prior)

            denom = total_tokens[label] + self.args.alpha * vocab_size
            unknown_log_likelihood[label] = math.log(self.args.alpha / denom)
            for token in vocab:
                prob = (token_counts[label][token] + self.args.alpha) / denom
                log_likelihood[label][token] = math.log(prob)

        return {
            "classes": classes,
            "vocabulary": vocab,
            "log_priors": log_priors,
            "log_likelihood": log_likelihood,
            "unknown_log_likelihood": unknown_log_likelihood,
        }

    @staticmethod
    def PredictLogScores(model: dict[str, object], tokens: list[str]) -> dict[str, float]:
        token_counts = Counter(tokens)
        scores = {}
        for label in model["classes"]:
            score = model["log_priors"][label]
            unknown_ll = model["unknown_log_likelihood"][label]
            likelihoods = model["log_likelihood"][label]
            for token, count in token_counts.items():
                token_ll = likelihoods[token] if token in likelihoods else unknown_ll
                score += count * token_ll
            scores[label] = score
        return scores

    @staticmethod
    def Softmax(log_scores: dict[str, float]) -> dict[str, float]:
        max_log = max(log_scores.values())
        exps = {label: math.exp(score - max_log) for label, score in log_scores.items()}
        total = sum(exps.values())
        return {label: value / total for label, value in exps.items()}

    def EvaluateModel(self, model: dict[str, object], examples: list[dict[str, object]]) -> dict[str, object]:
        classes = model["classes"]
        confusion = {true_label: {pred_label: 0 for pred_label in classes} for true_label in classes}
        correct = 0
        for ex in examples:
            label = ex["label"]
            scores = self.PredictLogScores(model, ex["tokens"])
            pred = max(scores.items(), key=lambda item: item[1])[0]
            confusion[label][pred] += 1
            if pred == label:
                correct += 1
        accuracy = correct / len(examples) if examples else 0.0
        return {"accuracy": accuracy, "confusion": confusion, "count": len(examples)}

    @staticmethod
    def WriteCsv(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def BuildPredictionRows(self, model: dict[str, object]) -> list[dict[str, str]]:
        prediction_rows = []
        for paper_id, paper in self.papers.items():
            abstract = self.NormalizeText(paper["Abstract"])
            if not abstract:
                continue
            tokens = self.FilterTokens(self.Tokenize(abstract))
            if not tokens:
                continue

            log_scores = self.PredictLogScores(model, tokens)
            probs = self.Softmax(log_scores)
            ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
            pred = ordered[0][0]
            top_prob = ordered[0][1]
            margin = ordered[0][1] - ordered[1][1]

            genes = self.links[paper_id]["genes"] if paper_id in self.links else []
            drugs = self.links[paper_id]["drugs"] if paper_id in self.links else []
            prediction_rows.append(
                {
                    "PaperID": paper_id,
                    "DOI": paper["DOI"],
                    "Biomarkers": "|".join(genes),
                    "Drugs": "|".join(drugs),
                    "PredictedStance": pred,
                    "P_supportive": f"{probs['supportive']:.6f}",
                    "P_opposing": f"{probs['opposing']:.6f}",
                    "P_neutral": f"{probs['neutral']:.6f}",
                    "ConfidenceScore": f"{top_prob:.6f}",
                    "ConfidenceMargin": f"{margin:.6f}",
                }
            )
        return prediction_rows

    def Run(self) -> None:
        self.EnsureResources()
        self.LoadPapers()
        self.LoadLinks()
        self.LoadWeakLabels()

        examples = self.BuildTrainingExamples()
        train_examples, valid_examples = self.SplitTrainValidation(examples)
        vocab = self.BuildVocab([ex["tokens"] for ex in train_examples])
        model = self.TrainMultinomialNB(train_examples, sorted(self.SUPPORTED_LABELS), vocab)

        train_eval = self.EvaluateModel(model, train_examples)
        valid_eval = self.EvaluateModel(model, valid_examples)
        prediction_rows = self.BuildPredictionRows(model)

        os.makedirs(self.args.output_dir, exist_ok=True)
        predictions_path = os.path.join(self.args.output_dir, "bow_stance_predictions.csv")
        eval_path = os.path.join(self.args.output_dir, "bow_stance_eval.json")

        self.WriteCsv(
            predictions_path,
            prediction_rows,
            [
                "PaperID",
                "DOI",
                "Biomarkers",
                "Drugs",
                "PredictedStance",
                "P_supportive",
                "P_opposing",
                "P_neutral",
                "ConfidenceScore",
                "ConfidenceMargin",
            ],
        )

        model_payload = {
            "model_type": "multinomial_nb_bow",
            "classes": model["classes"],
            "vocabulary_size": len(model["vocabulary"]),
            "vocabulary": model["vocabulary"],
            "log_priors": model["log_priors"],
            "log_likelihood": model["log_likelihood"],
            "unknown_log_likelihood": model["unknown_log_likelihood"],
            "params": {
                "alpha": self.args.alpha,
                "min_df": self.args.min_df,
                "max_vocab": self.args.max_vocab,
                "min_confidence": self.args.min_confidence,
                "drop_unclear": self.args.drop_unclear,
                "seed": self.args.seed,
            },
            "train_example_count": len(train_examples),
            "validation_example_count": len(valid_examples),
        }
        with open(self.args.model_output, "w", encoding="utf-8") as handle:
            json.dump(model_payload, handle)

        with open(eval_path, "w", encoding="utf-8") as handle:
            json.dump({"train": train_eval, "validation": valid_eval}, handle, indent=2)

        print(f"Training examples: {len(train_examples)}")
        print(f"Validation examples: {len(valid_examples)}")
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Train accuracy: {train_eval['accuracy']:.4f}")
        print(f"Validation accuracy: {valid_eval['accuracy']:.4f}")
        print(f"Model: {self.args.model_output}")
        print(f"Predictions: {predictions_path}")
        print(f"Evaluation: {eval_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BoW stance classifier and score papers")
    parser.add_argument("--papers", default="data/raw/papers.csv")
    parser.add_argument("--paper-sources", default="data/raw/paper_sources.csv")
    parser.add_argument("--weak-labels", default="output_csvs/traditional_stance_labels.csv")
    parser.add_argument("--labels-paperid-col", default="PaperID")
    parser.add_argument("--labels-label-col", default="PredictedStance")
    parser.add_argument("--labels-confidence-col", default="ConfidenceScore", help="probability-like column in [0,1]")
    parser.add_argument("--output-dir", default="output_csvs")
    parser.add_argument("--model-output", default="output_csvs/bow_stance_model.json")
    parser.add_argument("--min-confidence", type=float, default=0.60, help="minimum weak-label probability")
    parser.add_argument("--drop-unclear", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=410)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-vocab", type=int, default=40000)
    parser.add_argument("--debug-max-papers", type=int, default=0)
    parser.add_argument("--keep-numeric", action="store_true")
    return parser


if __name__ == "__main__":
    classifier = BoWStanceClassifier(build_arg_parser().parse_args())
    classifier.Run()
