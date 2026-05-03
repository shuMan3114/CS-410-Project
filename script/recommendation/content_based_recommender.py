#!/usr/bin/env python3
"""
Content-based paper recommender using organ/biomarker/drug/stance signals.

Inputs:
  - papers.csv
  - paper_sources.csv
  - biomarkers.csv (for Organ mapping)
  - final_paper_stances.csv (output of text-mining pipeline)
  - optional user feedback CSV

Output:
  - output_csvs/content_recommendations.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict


class ContentBasedRecommender:
    TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-+/#]*")

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.query_organs: set[str] = set()
        self.query_biomarkers: set[str] = set()
        self.query_drugs: set[str] = set()
        self.query_stances: set[str] = set()
        self.query_text_tokens: set[str] = set()

        self.papers: dict[str, dict[str, str]] = {}
        self.biomarker_meta: dict[str, dict[str, str]] = {}
        self.source_rows: list[dict[str, str]] = []
        self.stance_map: dict[str, dict[str, str]] = {}
        self.feedback_by_paper: dict[str, Counter] = {}
        self.feedback_state_by_user: dict[str, dict[str, Counter]] = {}
        self.runtime_feedback_events: list[dict[str, str]] = []
        self.paper_data: dict[str, dict[str, set[str]]] = {}
        self.interaction_pairs: set[tuple[str, str]] = set()
        self.structure_biomarker_ids: set[str] = set()
        self.last_recommendation_mode = "personalized"

    @staticmethod
    def NormalizeText(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    @classmethod
    def Tokenize(cls, text: str) -> list[str]:
        return [m.group(0).lower() for m in cls.TOKEN_RE.finditer(text)]

    @classmethod
    def NormalizeTerm(cls, value: str) -> str:
        return cls.NormalizeText(value).lower()

    @classmethod
    def ParseCsvList(cls, value: str) -> list[str]:
        if not value:
            return []
        return [cls.NormalizeTerm(part) for part in value.split(",") if cls.NormalizeTerm(part)]

    @classmethod
    def ParseDelimitedTerms(cls, value: str) -> list[str]:
        if not value:
            return []
        parts = re.split(r"[,\|;]", value)
        return [cls.NormalizeTerm(part) for part in parts if cls.NormalizeTerm(part)]

    @staticmethod
    def ParsePublicationYear(publication_date: str) -> int:
        year_match = re.search(r"(19|20)\d{2}", publication_date)
        return int(year_match.group(0)) if year_match else 0

    @staticmethod
    def Clamp01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    @classmethod
    def NormalizeStanceLabel(cls, value: str) -> str:
        normalized = cls.NormalizeTerm(value)
        mapping = {
            "positive": "supportive",
            "negative": "opposing",
            "inconclusive": "unclear",
        }
        if normalized in mapping:
            return mapping[normalized]
        return normalized

    @staticmethod
    def RequireColumns(fieldnames: list[str], required: list[str], csv_path: str) -> None:
        missing = [col for col in required if col not in fieldnames]
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {missing}")

    @staticmethod
    def RequireDictKeys(payload: dict, required: list[str], context: str) -> None:
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{context} missing required keys: {missing}")

    @classmethod
    def ReadCsvDicts(cls, path: str, required_cols: list[str]) -> list[dict[str, str]]:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            cls.RequireColumns(reader.fieldnames or [], required_cols, path)
            return list(reader)

    def LoadApiRequest(self) -> None:
        if not self.args.api_request_json:
            return
        with open(self.args.api_request_json, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("API request payload must be a JSON object.")

        if "user_id" in payload:
            if not isinstance(payload["user_id"], str):
                raise ValueError("API request field user_id must be a string.")
            self.args.user_id = payload["user_id"]

        if "top_k" in payload:
            self.args.top_k = int(payload["top_k"])
            if self.args.top_k <= 0:
                raise ValueError("API request field top_k must be > 0.")

        if "query" in payload:
            query = payload["query"]
            if not isinstance(query, dict):
                raise ValueError("API request field query must be an object.")
            if "organs" in query:
                if not isinstance(query["organs"], list):
                    raise ValueError("query.organs must be a list.")
                self.args.query_organs = ",".join(str(item) for item in query["organs"])
            if "biomarkers" in query:
                if not isinstance(query["biomarkers"], list):
                    raise ValueError("query.biomarkers must be a list.")
                self.args.query_biomarkers = ",".join(str(item) for item in query["biomarkers"])
            if "drugs" in query:
                if not isinstance(query["drugs"], list):
                    raise ValueError("query.drugs must be a list.")
                self.args.query_drugs = ",".join(str(item) for item in query["drugs"])
            if "stances" in query:
                if not isinstance(query["stances"], list):
                    raise ValueError("query.stances must be a list.")
                self.args.query_stances = ",".join(str(item) for item in query["stances"])
            self.args.query_text = str(query["text"]) if "text" in query else self.args.query_text
            if "year_start" in query:
                self.args.query_year_start = int(query["year_start"])
            if "year_end" in query:
                self.args.query_year_end = int(query["year_end"])

        if "weights" in payload:
            weights = payload["weights"]
            if not isinstance(weights, dict):
                raise ValueError("API request field weights must be an object.")
            if "organ" in weights:
                self.args.weight_organ = float(weights["organ"])
            if "biomarker" in weights:
                self.args.weight_biomarker = float(weights["biomarker"])
            if "drug" in weights:
                self.args.weight_drug = float(weights["drug"])
            if "stance" in weights:
                self.args.weight_stance = float(weights["stance"])
            if "text" in weights:
                self.args.weight_text = float(weights["text"])
            if "time" in weights:
                self.args.weight_time = float(weights["time"])
            if "structure" in weights:
                self.args.weight_structure = float(weights["structure"])
            if "interaction" in weights:
                self.args.weight_interaction = float(weights["interaction"])
            if "cold_start_confidence" in weights:
                self.args.cold_start_confidence_weight = float(weights["cold_start_confidence"])
            if "cold_start_recency" in weights:
                self.args.cold_start_recency_weight = float(weights["cold_start_recency"])
            if "cold_start_entity" in weights:
                self.args.cold_start_entity_weight = float(weights["cold_start_entity"])

        if "flags" in payload:
            flags = payload["flags"]
            if not isinstance(flags, dict):
                raise ValueError("API request field flags must be an object.")
            if "exclude_unclear" in flags:
                self.args.exclude_unclear = bool(flags["exclude_unclear"])
            if "require_all_filters" in flags:
                self.args.require_all_filters = bool(flags["require_all_filters"])
            if "respect_hide_feedback" in flags:
                self.args.respect_hide_feedback = bool(flags["respect_hide_feedback"])
            if "force_cold_start" in flags:
                self.args.force_cold_start = bool(flags["force_cold_start"])
            if "cold_start_diversify_by_stance" in flags:
                self.args.cold_start_diversify_by_stance = bool(flags["cold_start_diversify_by_stance"])

        if "feedback_events" in payload:
            events = payload["feedback_events"]
            if not isinstance(events, list):
                raise ValueError("API request field feedback_events must be a list.")
            normalized_events = []
            for idx, event in enumerate(events):
                if not isinstance(event, dict):
                    raise ValueError(f"feedback_events[{idx}] must be an object.")
                self.RequireDictKeys(event, ["paper_id", "feedback_type"], f"feedback_events[{idx}]")
                normalized_events.append(
                    {
                        "paper_id": str(event["paper_id"]),
                        "feedback_type": self.NormalizeTerm(str(event["feedback_type"])),
                    }
                )
            self.runtime_feedback_events = normalized_events

    def LoadFeedbackState(self) -> None:
        self.feedback_state_by_user = {}
        if not self.args.feedback_state_json:
            return
        if not os.path.exists(self.args.feedback_state_json):
            return
        with open(self.args.feedback_state_json, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Feedback state JSON must be an object keyed by user id.")
        user_state = {}
        for user_id, paper_map in payload.items():
            if not isinstance(paper_map, dict):
                raise ValueError(f"Feedback state for user {user_id} must be an object.")
            user_state[user_id] = {}
            for paper_id, counts in paper_map.items():
                if not isinstance(counts, dict):
                    raise ValueError(f"Feedback state for user {user_id}, paper {paper_id} must be an object.")
                counter = Counter()
                for feedback_type, count in counts.items():
                    counter[self.NormalizeTerm(str(feedback_type))] = int(count)
                user_state[user_id][paper_id] = counter
        self.feedback_state_by_user = user_state

    def MergeStateFeedbackForUser(self) -> None:
        if not self.args.user_id:
            return
        if self.args.user_id not in self.feedback_state_by_user:
            return
        for paper_id, counts in self.feedback_state_by_user[self.args.user_id].items():
            if paper_id not in self.feedback_by_paper:
                self.feedback_by_paper[paper_id] = Counter()
            self.feedback_by_paper[paper_id].update(counts)

    def ApplyRuntimeFeedbackEvents(self) -> None:
        if not self.runtime_feedback_events:
            return
        if not self.args.user_id:
            raise ValueError("API feedback events require a user_id.")
        if self.args.user_id not in self.feedback_state_by_user:
            self.feedback_state_by_user[self.args.user_id] = {}

        for event in self.runtime_feedback_events:
            paper_id = event["paper_id"]
            feedback_type = event["feedback_type"]
            if paper_id not in self.feedback_by_paper:
                self.feedback_by_paper[paper_id] = Counter()
            self.feedback_by_paper[paper_id][feedback_type] += 1

            if paper_id not in self.feedback_state_by_user[self.args.user_id]:
                self.feedback_state_by_user[self.args.user_id][paper_id] = Counter()
            self.feedback_state_by_user[self.args.user_id][paper_id][feedback_type] += 1

    def SaveFeedbackState(self) -> None:
        if not self.args.feedback_state_json:
            return
        serialized = {}
        for user_id, paper_map in self.feedback_state_by_user.items():
            serialized[user_id] = {}
            for paper_id, counts in paper_map.items():
                serialized[user_id][paper_id] = dict(counts)
        with open(self.args.feedback_state_json, "w", encoding="utf-8") as handle:
            json.dump(serialized, handle, indent=2)

    @staticmethod
    def ToApiRecommendations(recommendations: list[dict[str, str]]) -> list[dict[str, object]]:
        api_rows = []
        for rank, row in enumerate(recommendations, start=1):
            api_rows.append(
                {
                    "rank": rank,
                    "paper_id": row["PaperID"],
                    "doi": row["DOI"],
                    "title": row["Title"],
                    "predicted_stance": row["PredictedStance"],
                    "publication_year": int(row["PublicationYear"]),
                    "stance_confidence": float(row["StanceConfidence"]),
                    "final_score": float(row["FinalScore"]),
                    "base_score": float(row["BaseScore"]),
                    "time_score": float(row["TimeScore"]),
                    "structure_score": float(row["StructureScore"]),
                    "interaction_score": float(row["InteractionScore"]),
                    "feedback_delta": float(row["FeedbackDelta"]),
                    "recommendation_mode": row["RecommendationMode"],
                    "cold_start_confidence_score": float(row["ColdStartConfidenceScore"]),
                    "cold_start_recency_score": float(row["ColdStartRecencyScore"]),
                    "cold_start_entity_score": float(row["ColdStartEntityScore"]),
                    "organs": row["Organs"].split("|") if row["Organs"] else [],
                    "biomarkers": row["Biomarkers"].split("|") if row["Biomarkers"] else [],
                    "drugs": row["Drugs"].split("|") if row["Drugs"] else [],
                    "interaction_match_count": int(row["InteractionMatchCount"]),
                    "structure_biomarker_count": int(row["StructureBiomarkerCount"]),
                    "matched_organs": row["MatchedOrgans"].split("|") if row["MatchedOrgans"] else [],
                    "matched_biomarkers": row["MatchedBiomarkers"].split("|") if row["MatchedBiomarkers"] else [],
                    "matched_drugs": row["MatchedDrugs"].split("|") if row["MatchedDrugs"] else [],
                    "matched_stance": row["MatchedStance"],
                    "matched_query_terms": row["MatchedQueryTerms"].split("|") if row["MatchedQueryTerms"] else [],
                }
            )
        return api_rows

    def WriteApiResponse(self, recommendations: list[dict[str, str]]) -> None:
        if not self.args.api_response_json:
            return
        payload = {
            "user_id": self.args.user_id,
            "top_k": self.args.top_k,
            "count": len(recommendations),
            "mode": self.last_recommendation_mode,
            "recommendations": self.ToApiRecommendations(recommendations),
        }
        with open(self.args.api_response_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def LoadPapers(self) -> None:
        rows = self.ReadCsvDicts(
            self.args.papers,
            ["PaperID", "Title", "DOI", "PublicationDate", "Abstract"],
        )
        self.papers = {row["PaperID"]: row for row in rows}

    def LoadBiomarkerMeta(self) -> None:
        rows = self.ReadCsvDicts(self.args.biomarkers, ["BiomarkerID", "Name", "Organ", "GeneSymbol"])
        self.biomarker_meta = {row["BiomarkerID"]: row for row in rows}

    def LoadPaperSources(self) -> None:
        self.source_rows = self.ReadCsvDicts(
            self.args.paper_sources,
            ["PaperID", "BiomarkerID", "GeneSymbol", "DrugID", "DrugName"],
        )

    def LoadStance(self) -> None:
        rows = self.ReadCsvDicts(
            self.args.stance,
            ["PaperID", "FinalLabel", "FinalConfidence"],
        )
        stance = {}
        for row in rows:
            confidence = float(row["FinalConfidence"])
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(
                    f"{self.args.stance} has out-of-range stance confidence for "
                    f"PaperID={row['PaperID']}: {confidence}. Expected [0,1]."
                )
            stance[row["PaperID"]] = {
                "label": self.NormalizeStanceLabel(row["FinalLabel"]),
                "confidence": confidence,
            }
        self.stance_map = stance

    def LoadDrugGeneInteractions(self) -> None:
        rows = self.ReadCsvDicts(
            self.args.drug_gene_interactions,
            ["GeneSymbol", "DrugName"],
        )
        pairs: set[tuple[str, str]] = set()
        for row in rows:
            gene = self.NormalizeTerm(row["GeneSymbol"])
            drug = self.NormalizeTerm(row["DrugName"])
            if gene and drug:
                pairs.add((gene, drug))
        self.interaction_pairs = pairs

    def LoadBiomarkerStructureMap(self) -> None:
        rows = self.ReadCsvDicts(
            self.args.biomarker_uniprot_map,
            ["BiomarkerID", "UniProtAccession"],
        )
        mapped_ids = set()
        for row in rows:
            if self.NormalizeTerm(row["UniProtAccession"]):
                mapped_ids.add(row["BiomarkerID"])
        self.structure_biomarker_ids = mapped_ids

    def LoadFeedback(self) -> None:
        self.feedback_by_paper = defaultdict(Counter)
        if not self.args.feedback_csv:
            return
        if not self.args.user_id:
            raise ValueError("If --feedback-csv is provided, --user-id is required.")

        rows = self.ReadCsvDicts(
            self.args.feedback_csv,
            [self.args.feedback_user_col, self.args.feedback_paperid_col, self.args.feedback_type_col],
        )
        feedback = defaultdict(Counter)
        for row in rows:
            if row[self.args.feedback_user_col] != self.args.user_id:
                continue
            feedback_type = self.NormalizeTerm(row[self.args.feedback_type_col])
            feedback[row[self.args.feedback_paperid_col]][feedback_type] += 1
        self.feedback_by_paper = feedback

    @staticmethod
    def FeedbackAdjustment(
        counts: Counter,
        positive_boost: float,
        negative_penalty: float,
    ) -> tuple[float, bool]:
        positive_terms = {"positive", "like", "liked", "save", "saved", "upvote"}
        negative_terms = {"negative", "dislike", "disliked", "downvote"}
        hide_terms = {"hide", "hidden", "block", "blocked", "skip", "skipped"}

        has_hide = any(term in counts for term in hide_terms)
        pos_count = sum(count for term, count in counts.items() if term in positive_terms)
        neg_count = sum(count for term, count in counts.items() if term in negative_terms)
        delta = positive_boost * pos_count - negative_penalty * neg_count
        return delta, has_hide

    def SetupQuery(self) -> None:
        if self.args.query_year_start > 0 and self.args.query_year_end > 0:
            if self.args.query_year_start > self.args.query_year_end:
                raise ValueError("--query-year-start cannot be greater than --query-year-end.")
        self.query_organs = set(self.ParseCsvList(self.args.query_organs))
        self.query_biomarkers = set(self.ParseCsvList(self.args.query_biomarkers))
        self.query_drugs = set(self.ParseCsvList(self.args.query_drugs))
        self.query_stances = set(self.ParseCsvList(self.args.query_stances))
        self.query_text_tokens = set(self.Tokenize(self.NormalizeText(self.args.query_text)))

    def BuildPaperData(self) -> None:
        paper_data: dict[str, dict[str, set[str]]] = {}
        for row in self.source_rows:
            paper_id = row["PaperID"]
            if paper_id not in self.papers:
                continue
            if paper_id not in paper_data:
                paper_data[paper_id] = {
                    "biomarker_ids": set(),
                    "biomarker_names": set(),
                    "gene_symbols": set(),
                    "organs": set(),
                    "drug_ids": set(),
                    "drug_names": set(),
                }
            info = paper_data[paper_id]
            biomarker_id = row["BiomarkerID"]
            drug_id = row["DrugID"]
            gene_symbol = self.NormalizeTerm(row["GeneSymbol"])
            drug_name = self.NormalizeTerm(row["DrugName"])
            info["biomarker_ids"].add(biomarker_id)
            info["gene_symbols"].add(gene_symbol)
            info["drug_ids"].add(drug_id)
            info["drug_names"].add(drug_name)

            if biomarker_id in self.biomarker_meta:
                bm = self.biomarker_meta[biomarker_id]
                if self.NormalizeTerm(bm["Name"]):
                    info["biomarker_names"].add(self.NormalizeTerm(bm["Name"]))
                if self.NormalizeTerm(bm["GeneSymbol"]):
                    info["gene_symbols"].add(self.NormalizeTerm(bm["GeneSymbol"]))
                for organ_term in self.ParseDelimitedTerms(bm["Organ"]):
                    info["organs"].add(organ_term)
        self.paper_data = paper_data

    def IsColdStart(self, query_filter_count: int) -> bool:
        if self.args.force_cold_start:
            return True
        has_query = bool(
            query_filter_count > 0
            or (self.args.query_year_start > 0 and self.args.query_year_end > 0)
        )
        if has_query:
            return False
        has_feedback = any(sum(counts.values()) > 0 for counts in self.feedback_by_paper.values())
        if has_feedback:
            return False
        return True

    def DiversifyByStance(self, ranked_rows: list[dict[str, str]]) -> list[dict[str, str]]:
        stance_order = ["supportive", "opposing", "neutral", "unclear"]
        buckets = defaultdict(list)
        for row in ranked_rows:
            buckets[row["PredictedStance"]].append(row)

        diversified = []
        while len(diversified) < self.args.top_k:
            added_in_round = False
            for stance in stance_order:
                if buckets[stance]:
                    diversified.append(buckets[stance].pop(0))
                    added_in_round = True
                    if len(diversified) >= self.args.top_k:
                        break
            if not added_in_round:
                break

        remaining = []
        for rows in buckets.values():
            remaining.extend(rows)
        if remaining and len(diversified) < self.args.top_k:
            remaining.sort(key=lambda row: float(row["FinalScore"]), reverse=True)
            take_n = self.args.top_k - len(diversified)
            diversified.extend(remaining[:take_n])
        return diversified

    def BuildRecommendations(self) -> list[dict[str, str]]:
        query_filter_count = sum(
            [
                1 if self.query_organs else 0,
                1 if self.query_biomarkers else 0,
                1 if self.query_drugs else 0,
                1 if self.query_stances else 0,
                1 if self.query_text_tokens else 0,
            ]
        )

        recommendations = []
        paper_items = list(self.paper_data.items())
        if self.args.debug_max_papers > 0:
            paper_items = paper_items[: self.args.debug_max_papers]

        cold_start_mode = self.IsColdStart(query_filter_count)
        self.last_recommendation_mode = "cold_start" if cold_start_mode else "personalized"

        candidate_years = []
        for paper_id, _ in paper_items:
            publication_year = self.ParsePublicationYear(self.papers[paper_id]["PublicationDate"])
            if publication_year > 0:
                candidate_years.append(publication_year)
        min_year = min(candidate_years) if candidate_years else 0
        max_year = max(candidate_years) if candidate_years else 0
        year_range = max_year - min_year

        for paper_id, info in paper_items:
            if paper_id not in self.stance_map:
                continue
            stance_label = self.stance_map[paper_id]["label"]
            stance_conf = self.stance_map[paper_id]["confidence"]

            if self.args.exclude_unclear and stance_label == "unclear":
                continue

            title = self.NormalizeText(self.papers[paper_id]["Title"])
            abstract = self.NormalizeText(self.papers[paper_id]["Abstract"])
            text_tokens = set(self.Tokenize(f"{title} {abstract}"))
            publication_date = self.papers[paper_id]["PublicationDate"]
            publication_year = self.ParsePublicationYear(publication_date)

            organ_hits = sorted(self.query_organs.intersection(info["organs"]))
            biomarker_hits = sorted(
                self.query_biomarkers.intersection(info["gene_symbols"].union(info["biomarker_names"]))
            )
            drug_hits = sorted(self.query_drugs.intersection(info["drug_names"]))
            stance_hits = [stance_label] if self.query_stances and stance_label in self.query_stances else []
            text_hits = sorted(self.query_text_tokens.intersection(text_tokens))

            interaction_match_count = 0
            for gene in info["gene_symbols"]:
                for drug in info["drug_names"]:
                    if (gene, drug) in self.interaction_pairs:
                        interaction_match_count += 1

            structure_biomarker_count = 0
            for biomarker_id in info["biomarker_ids"]:
                has_map_structure = biomarker_id in self.structure_biomarker_ids
                has_alphafold_id = False
                if biomarker_id in self.biomarker_meta:
                    has_alphafold_id = bool(self.NormalizeTerm(self.biomarker_meta[biomarker_id]["AlphaFoldID"]))
                if has_map_structure or has_alphafold_id:
                    structure_biomarker_count += 1

            match_flags = [
                bool(organ_hits) if self.query_organs else False,
                bool(biomarker_hits) if self.query_biomarkers else False,
                bool(drug_hits) if self.query_drugs else False,
                bool(stance_hits) if self.query_stances else False,
                bool(text_hits) if self.query_text_tokens else False,
            ]
            matched_filter_count = sum(1 for flag in match_flags if flag)
            if query_filter_count > 0 and matched_filter_count == 0:
                continue
            if self.args.require_all_filters:
                if self.query_organs and not organ_hits:
                    continue
                if self.query_biomarkers and not biomarker_hits:
                    continue
                if self.query_drugs and not drug_hits:
                    continue
                if self.query_stances and not stance_hits:
                    continue
                if self.query_text_tokens and not text_hits:
                    continue

            organ_score = self.args.weight_organ * (len(organ_hits) / len(self.query_organs)) if self.query_organs else 0.0
            biomarker_score = (
                self.args.weight_biomarker * (len(biomarker_hits) / len(self.query_biomarkers))
                if self.query_biomarkers
                else 0.0
            )
            drug_score = self.args.weight_drug * (len(drug_hits) / len(self.query_drugs)) if self.query_drugs else 0.0
            stance_score = self.args.weight_stance if stance_hits else 0.0
            text_score = self.args.weight_text * (len(text_hits) / len(self.query_text_tokens)) if self.query_text_tokens else 0.0
            interaction_score = 0.0
            if self.args.weight_interaction > 0:
                interaction_score = self.args.weight_interaction * min(1.0, interaction_match_count / 3.0)

            structure_score = 0.0
            if self.args.weight_structure > 0 and len(info["biomarker_ids"]) > 0:
                structure_score = self.args.weight_structure * (structure_biomarker_count / len(info["biomarker_ids"]))

            time_score = 0.0
            if self.args.query_year_start > 0 and self.args.query_year_end > 0 and publication_year > 0:
                if self.args.query_year_start <= publication_year <= self.args.query_year_end:
                    time_score = self.args.weight_time
                else:
                    dist = min(
                        abs(publication_year - self.args.query_year_start),
                        abs(publication_year - self.args.query_year_end),
                    )
                    time_score = self.args.weight_time / (1.0 + float(dist))

            cold_start_confidence_score = 0.0
            cold_start_recency_score = 0.0
            cold_start_entity_score = 0.0
            recommendation_mode = "personalized"

            if cold_start_mode:
                recommendation_mode = "cold_start"
                if year_range > 0 and publication_year > 0:
                    recency_norm = (publication_year - min_year) / year_range
                elif publication_year > 0:
                    recency_norm = 1.0
                else:
                    recency_norm = 0.0
                recency_norm = self.Clamp01(recency_norm)

                entity_density = self.Clamp01(
                    (len(info["gene_symbols"]) + len(info["drug_names"]) + len(info["organs"])) / 10.0
                )

                cold_start_confidence_score = self.args.cold_start_confidence_weight * stance_conf
                cold_start_recency_score = self.args.cold_start_recency_weight * recency_norm
                cold_start_entity_score = self.args.cold_start_entity_weight * entity_density

                base_score = (
                    cold_start_confidence_score
                    + cold_start_recency_score
                    + cold_start_entity_score
                    + interaction_score
                    + structure_score
                )
            else:
                base_score = (
                    organ_score
                    + biomarker_score
                    + drug_score
                    + stance_score
                    + text_score
                    + interaction_score
                    + structure_score
                    + time_score
                )
                base_score += 0.75 * stance_conf

            feedback_delta = 0.0
            hide_due_to_feedback = False
            if paper_id in self.feedback_by_paper:
                feedback_delta, hide_due_to_feedback = self.FeedbackAdjustment(
                    self.feedback_by_paper[paper_id],
                    self.args.positive_feedback_boost,
                    self.args.negative_feedback_penalty,
                )
            if self.args.respect_hide_feedback and hide_due_to_feedback:
                continue

            final_score = base_score + feedback_delta
            recommendations.append(
                {
                    "PaperID": paper_id,
                    "DOI": self.papers[paper_id]["DOI"],
                    "Title": title,
                    "PredictedStance": stance_label,
                    "PublicationYear": str(publication_year),
                    "StanceConfidence": f"{stance_conf:.6f}",
                    "FinalScore": f"{final_score:.6f}",
                    "BaseScore": f"{base_score:.6f}",
                    "TimeScore": f"{time_score:.6f}",
                    "StructureScore": f"{structure_score:.6f}",
                    "InteractionScore": f"{interaction_score:.6f}",
                    "FeedbackDelta": f"{feedback_delta:.6f}",
                    "Organs": "|".join(sorted(info["organs"])),
                    "Biomarkers": "|".join(sorted(info["gene_symbols"])),
                    "Drugs": "|".join(sorted(info["drug_names"])),
                    "InteractionMatchCount": str(interaction_match_count),
                    "StructureBiomarkerCount": str(structure_biomarker_count),
                    "MatchedOrgans": "|".join(organ_hits),
                    "MatchedBiomarkers": "|".join(biomarker_hits),
                    "MatchedDrugs": "|".join(drug_hits),
                    "MatchedStance": "|".join(stance_hits),
                    "MatchedQueryTerms": "|".join(text_hits),
                    "RecommendationMode": recommendation_mode,
                    "ColdStartConfidenceScore": f"{cold_start_confidence_score:.6f}",
                    "ColdStartRecencyScore": f"{cold_start_recency_score:.6f}",
                    "ColdStartEntityScore": f"{cold_start_entity_score:.6f}",
                }
            )

        recommendations.sort(key=lambda row: float(row["FinalScore"]), reverse=True)
        if cold_start_mode and self.args.cold_start_diversify_by_stance:
            recommendations = self.DiversifyByStance(recommendations)
        return recommendations[: self.args.top_k]

    def WriteOutput(self, recommendations: list[dict[str, str]]) -> None:
        output_dir = os.path.dirname(self.args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "PaperID",
                    "DOI",
                    "Title",
                    "PredictedStance",
                    "PublicationYear",
                    "StanceConfidence",
                    "FinalScore",
                    "BaseScore",
                    "TimeScore",
                    "StructureScore",
                    "InteractionScore",
                    "FeedbackDelta",
                    "Organs",
                    "Biomarkers",
                    "Drugs",
                    "InteractionMatchCount",
                    "StructureBiomarkerCount",
                    "MatchedOrgans",
                    "MatchedBiomarkers",
                    "MatchedDrugs",
                    "MatchedStance",
                    "MatchedQueryTerms",
                    "RecommendationMode",
                    "ColdStartConfidenceScore",
                    "ColdStartRecencyScore",
                    "ColdStartEntityScore",
                ],
            )
            writer.writeheader()
            writer.writerows(recommendations)

    def Run(self) -> None:
        self.LoadApiRequest()
        self.SetupQuery()
        self.LoadPapers()
        self.LoadBiomarkerMeta()
        self.LoadPaperSources()
        self.LoadStance()
        self.LoadDrugGeneInteractions()
        self.LoadBiomarkerStructureMap()
        self.LoadFeedbackState()
        self.LoadFeedback()
        self.MergeStateFeedbackForUser()
        self.ApplyRuntimeFeedbackEvents()
        self.BuildPaperData()

        recommendations = self.BuildRecommendations()
        if not self.args.disable_csv_output:
            self.WriteOutput(recommendations)
        self.WriteApiResponse(recommendations)
        self.SaveFeedbackState()

        if not self.args.disable_csv_output:
            print(f"Wrote {len(recommendations)} recommendations to {self.args.output}")
        if self.args.api_response_json:
            print(f"Wrote API response JSON: {self.args.api_response_json}")
        if recommendations:
            print("Top 5 papers:")
            for row in recommendations[:5]:
                print(
                    f"  PaperID={row['PaperID']} score={row['FinalScore']} "
                    f"stance={row['PredictedStance']} title={row['Title'][:90]}"
                )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Content-based paper recommender with user feedback reranking")
    parser.add_argument("--papers", default="data/raw/papers.csv")
    parser.add_argument("--paper-sources", default="data/raw/paper_sources.csv")
    parser.add_argument("--biomarkers", default="data/raw/biomarkers.csv")
    parser.add_argument("--drug-gene-interactions", default="data/raw/drug_gene_interactions.csv")
    parser.add_argument("--biomarker-uniprot-map", default="data/raw/biomarker_uniprot_map.csv")
    parser.add_argument("--stance", default="output_csvs/final_paper_stances.csv")
    parser.add_argument("--feedback-csv", default="")
    parser.add_argument("--user-id", default="")
    parser.add_argument("--feedback-user-col", default="UserID")
    parser.add_argument("--feedback-paperid-col", default="PaperID")
    parser.add_argument("--feedback-type-col", default="AnnotationType")
    parser.add_argument(
        "--feedback-state-json",
        default="output_csvs/recommender_feedback_state.json",
        help="persistent feedback state for repeated API calls",
    )
    parser.add_argument("--api-request-json", default="", help="JSON request payload for frontend/backend integration")
    parser.add_argument("--api-response-json", default="", help="JSON response output path")
    parser.add_argument("--disable-csv-output", action="store_true")
    parser.add_argument("--output", default="output_csvs/content_recommendations.csv")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--debug-max-papers", type=int, default=0, help="0 means all candidate papers")
    parser.add_argument("--query-organs", default="")
    parser.add_argument("--query-biomarkers", default="")
    parser.add_argument("--query-drugs", default="")
    parser.add_argument("--query-stances", default="")
    parser.add_argument("--query-text", default="")
    parser.add_argument("--query-year-start", type=int, default=0)
    parser.add_argument("--query-year-end", type=int, default=0)
    parser.add_argument("--weight-organ", type=float, default=3.0)
    parser.add_argument("--weight-biomarker", type=float, default=4.0)
    parser.add_argument("--weight-drug", type=float, default=4.0)
    parser.add_argument("--weight-stance", type=float, default=2.5)
    parser.add_argument("--weight-text", type=float, default=1.2)
    parser.add_argument("--weight-time", type=float, default=1.0)
    parser.add_argument("--weight-structure", type=float, default=1.0)
    parser.add_argument("--weight-interaction", type=float, default=1.0)
    parser.add_argument("--cold-start-confidence-weight", type=float, default=2.0)
    parser.add_argument("--cold-start-recency-weight", type=float, default=2.0)
    parser.add_argument("--cold-start-entity-weight", type=float, default=1.5)
    parser.add_argument("--positive-feedback-boost", type=float, default=2.0)
    parser.add_argument("--negative-feedback-penalty", type=float, default=2.5)
    parser.add_argument("--respect-hide-feedback", action="store_true")
    parser.add_argument("--exclude-unclear", action="store_true")
    parser.add_argument("--require-all-filters", action="store_true")
    parser.add_argument("--force-cold-start", action="store_true")
    parser.add_argument(
        "--disable-cold-start-diversify-by-stance",
        action="store_false",
        dest="cold_start_diversify_by_stance",
    )
    parser.set_defaults(cold_start_diversify_by_stance=True)
    return parser


if __name__ == "__main__":
    recommender = ContentBasedRecommender(build_arg_parser().parse_args())
    recommender.Run()
