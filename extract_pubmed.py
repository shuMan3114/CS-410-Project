#!/usr/bin/env python3
"""
ProtEvidenceDB — PubMed Paper Extraction Pipeline (v2)
========================================================
Fetches papers from PubMed with SMART biomarker↔drug paired queries so
every paper can be linked back to a specific (biomarker, drug) pair for
stance labelling.

Query Tiers (all maintain biomarker↔drug pairing):
  T1 — gene_symbol AND drug_name                       (strict, high precision)
  T2 — (gene_symbol OR synonyms OR protein_name) AND drug_name
                                                        (relaxed gene matching)
  T3 — gene_terms AND drug_mechanism AND "cancer"       (when drug name is obscure)
  T4 — organ_cancer AND gene_terms AND drug_name        (adds organ for context)
  T5 — organ_cancer AND gene_terms AND ("therapy" OR "treatment")
                                                        (broadest, still gene+organ)

Each paper result is tagged with the (BiomarkerID, DrugID) that generated
the query — this feeds directly into stances.csv.

Prerequisites (CSVs in DATA_DIR):
    - biomarkers.csv
    - drug_gene_interactions.csv
    - uniprot_proteins.csv         (for gene synonyms + protein names)
    - biomarker_uniprot_map.csv    (for biomarker → protein linking)

Output:
    - papers.csv           (all papers)
    - paper_sources.csv    (which biomarker+drug query found each paper)

Usage:
    python3 extract_pubmed.py
    python3 extract_pubmed.py --data-dir ./output_csvs --api-key YOUR_KEY
    python3 extract_pubmed.py --max-t1 99999 --max-t2 5000 --max-t5 500
"""

import requests
import csv
import time
import os
import sys
import re
import json
import argparse
from collections import defaultdict
from xml.etree import ElementTree as ET


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_DATA_DIR = "./output_csvs"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

PAPER_FIELDS = [
    "PaperID", "Title", "DOI", "PublicationDate",
    "JournalName", "JournalISSN", "PublicationTypes", "Abstract",
]

SOURCE_FIELDS = [
    "PaperID", "BiomarkerID", "DrugID", "GeneSymbol", "DrugName",
    "QueryTier", "QueryString",
]

ORGAN_MAP = {
    "prostate":      "prostate cancer",
    "breast":        "breast cancer",
    "lung":          "lung cancer OR NSCLC OR SCLC",
    "ovary":         "ovarian cancer",
    "pancreas":      "pancreatic cancer OR PDAC",
    "colon":         "colorectal cancer OR CRC",
    "bladder":       "bladder cancer OR urothelial",
    "liver":         "hepatocellular carcinoma OR liver cancer",
    "kidney":        "renal cell carcinoma OR kidney cancer",
    "brain":         "glioblastoma OR glioma OR brain cancer",
    "skin":          "melanoma OR skin cancer",
    "thyroid":       "thyroid cancer",
    "stomach":       "gastric cancer",
    "esophagus":     "esophageal cancer",
    "cervix":        "cervical cancer",
    "uterus":        "endometrial cancer",
    "head and neck": "head and neck squamous cell carcinoma OR HNSCC",
    "lymph node":    "lymphoma",
    "bone marrow":   "leukemia OR myeloma",
    "blood":         "leukemia OR lymphoma",
    "rectum":        "colorectal cancer OR rectal cancer",
}

# Drug mechanism → PubMed-friendly class term
MECHANISM_MAP = {
    "inhibitor":          "inhibitor",
    "agonist":            "agonist",
    "antagonist":         "antagonist",
    "blocker":            "blocker",
    "antibody":           "monoclonal antibody",
    "modulator":          "modulator",
    "activator":          "activator",
    "vaccine":            "vaccine",
    "negative modulator": "inhibitor",
}


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_all_data(data_dir):
    """Load and index all input CSVs."""
    print("Loading input CSVs …")

    biomarkers = load_csv(os.path.join(data_dir, "biomarkers.csv"))
    interactions = load_csv(os.path.join(data_dir, "drug_gene_interactions.csv"))
    proteins = load_csv(os.path.join(data_dir, "uniprot_proteins.csv"))
    bm_map = load_csv(os.path.join(data_dir, "biomarker_uniprot_map.csv"))

    print(f"  Biomarkers:    {len(biomarkers)}")
    print(f"  Interactions:  {len(interactions)}")
    print(f"  Proteins:      {len(proteins)}")
    print(f"  BM↔UniProt:   {len(bm_map)}")

    # ── Index: gene → BiomarkerID ──
    gene_to_bmid = {}
    gene_to_organ = {}
    for bm in biomarkers:
        gene = bm.get("GeneSymbol", "").strip()
        if gene:
            gene_to_bmid[gene] = bm["BiomarkerID"]
            gene_to_organ[gene] = bm.get("Organ", "").strip()

    # ── Index: gene → synonyms + protein name (from UniProt) ──
    gene_synonyms = {}     # gene → [syn1, syn2, ...]
    gene_protein_name = {} # gene → "Epidermal growth factor receptor"

    for prot in proteins:
        gene = prot.get("GeneSymbol", "").strip()
        if not gene:
            continue
        syns_raw = prot.get("GeneSynonyms", "").strip()
        if syns_raw:
            syns = [s.strip() for s in syns_raw.split("|") if s.strip()]
            gene_synonyms[gene] = syns
        pname = prot.get("ProteinName", "").strip()
        if pname and len(pname) > 5:
            gene_protein_name[gene] = pname

    # ── Index: gene → [(drug_id, drug_name, mechanism)] ──
    gene_drug_pairs = defaultdict(list)
    drug_id_to_name = {}

    for ix in interactions:
        gene = ix.get("GeneSymbol", "").strip()
        drug_name = ix.get("DrugName", "").strip()
        drug_id = ix.get("DrugID", "").strip()
        mechanism = ix.get("InteractionType", "").strip()

        if not gene or not drug_name:
            continue
        # Skip ChEMBL IDs and junk names
        if (drug_name.upper().startswith("CHEMBL") or
            drug_name.startswith("DB") and drug_name[2:7].isdigit() or
            len(drug_name) < 3 or not drug_name[0].isalpha()):
            continue

        gene_drug_pairs[gene].append({
            "drug_id": drug_id,
            "drug_name": drug_name,
            "mechanism": mechanism,
        })
        drug_id_to_name[drug_id] = drug_name

    print(f"  Genes with drug pairs: {len(gene_drug_pairs)}")
    print(f"  Genes with synonyms:   {len(gene_synonyms)}")
    print(f"  Genes with prot names: {len(gene_protein_name)}")

    return {
        "gene_to_bmid": gene_to_bmid,
        "gene_to_organ": gene_to_organ,
        "gene_synonyms": gene_synonyms,
        "gene_protein_name": gene_protein_name,
        "gene_drug_pairs": gene_drug_pairs,
        "drug_id_to_name": drug_id_to_name,
    }


# ─────────────────────────────────────────────
# QUERY BUILDING
# ─────────────────────────────────────────────
def build_gene_term(gene, synonyms, protein_name, expanded=False):
    """
    Build a PubMed gene search term.
    Basic:    "EGFR"
    Expanded: ("EGFR" OR "ERBB1" OR "HER1" OR "Epidermal growth factor receptor")
    """
    if not expanded:
        return f'"{gene}"'

    terms = [f'"{gene}"']
    for syn in (synonyms or []):
        if len(syn) >= 2 and syn != gene:
            terms.append(f'"{syn}"')
    if protein_name and len(protein_name) > 8:
        # Shorten very long protein names to avoid PubMed choking
        short = protein_name[:80]
        terms.append(f'"{short}"')

    if len(terms) == 1:
        return terms[0]
    return f'({" OR ".join(terms)})'


def organ_term(organ_raw):
    if not organ_raw:
        return None
    organ = organ_raw.split(",")[0].strip().lower()
    return ORGAN_MAP.get(organ, f"{organ} cancer" if organ and len(organ) > 2 else None)


def mechanism_term(mechanism_raw):
    """Convert DGIdb mechanism to PubMed search term."""
    if not mechanism_raw:
        return None
    # Take first mechanism if multi-valued
    mech = mechanism_raw.split(";")[0].strip().lower()
    return MECHANISM_MAP.get(mech)


def build_queries(data, max_t1, max_t2, max_t3, max_t4, max_t5):
    """
    Build tiered queries. Every query preserves the (gene, drug) pair
    so results can be traced back for stance labelling.

    Returns: [(tier, query_str, gene, drug_id, drug_name, bm_id)]
    """
    queries = []
    seen = set()  # (tier, gene, drug_name) dedup

    gene_drug_pairs = data["gene_drug_pairs"]
    gene_to_bmid = data["gene_to_bmid"]
    gene_to_organ = data["gene_to_organ"]
    gene_synonyms = data["gene_synonyms"]
    gene_protein_name = data["gene_protein_name"]

    counts = {"T1": 0, "T2": 0, "T3": 0, "T4": 0, "T5": 0}

    for gene in sorted(gene_drug_pairs.keys()):
        bm_id = gene_to_bmid.get(gene, "")
        organ = gene_to_organ.get(gene, "")
        syns = gene_synonyms.get(gene, [])
        pname = gene_protein_name.get(gene, "")
        cancer = organ_term(organ)

        gene_basic = build_gene_term(gene, None, None, expanded=False)
        gene_expanded = build_gene_term(gene, syns, pname, expanded=True)

        for entry in gene_drug_pairs[gene]:
            drug_id = entry["drug_id"]
            drug_name = entry["drug_name"]
            mech = entry["mechanism"]
            mech_term = mechanism_term(mech)

            # ── T1: gene + drug (simplest, most precise) ──
            if counts["T1"] < max_t1:
                key = ("T1", gene, drug_name)
                if key not in seen:
                    seen.add(key)
                    q = f'{gene_basic}[Title/Abstract] AND "{drug_name}"[Title/Abstract]'
                    queries.append(("T1", q, gene, drug_id, drug_name, bm_id))
                    counts["T1"] += 1

            # ── T2: (gene OR synonyms OR protein_name) + drug ──
            if counts["T2"] < max_t2 and gene_expanded != gene_basic:
                key = ("T2", gene, drug_name)
                if key not in seen:
                    seen.add(key)
                    q = f'{gene_expanded}[Title/Abstract] AND "{drug_name}"[Title/Abstract]'
                    queries.append(("T2", q, gene, drug_id, drug_name, bm_id))
                    counts["T2"] += 1

            # ── T3: gene + drug_mechanism + cancer (when drug name is obscure) ──
            if counts["T3"] < max_t3 and mech_term:
                key = ("T3", gene, mech_term)
                if key not in seen:
                    seen.add(key)
                    q = (f'{gene_expanded}[Title/Abstract] AND '
                         f'"{mech_term}"[Title/Abstract] AND '
                         f'cancer[Title/Abstract]')
                    queries.append(("T3", q, gene, drug_id, drug_name, bm_id))
                    counts["T3"] += 1

            # ── T4: organ + gene + drug (adds organ context) ──
            if counts["T4"] < max_t4 and cancer:
                key = ("T4", gene, drug_name, cancer)
                if key not in seen:
                    seen.add(key)
                    q = (f'({cancer})[Title/Abstract] AND '
                         f'{gene_basic}[Title/Abstract] AND '
                         f'"{drug_name}"[Title/Abstract]')
                    queries.append(("T4", q, gene, drug_id, drug_name, bm_id))
                    counts["T4"] += 1

        # ── T5: organ + gene + generic therapy (per gene, not per drug) ──
        if counts["T5"] < max_t5 and cancer:
            key = ("T5", gene, cancer)
            if key not in seen:
                seen.add(key)
                q = (f'({cancer})[Title/Abstract] AND '
                     f'{gene_expanded}[Title/Abstract] AND '
                     f'("drug" OR "therapy" OR "inhibitor" OR "treatment")[Title/Abstract]')
                # Use first drug pair for linking (T5 is broad)
                first = gene_drug_pairs[gene][0]
                queries.append(("T5", q, gene, first["drug_id"],
                                first["drug_name"], bm_id))
                counts["T5"] += 1

    return queries, counts


# ─────────────────────────────────────────────
# PubMed API
# ─────────────────────────────────────────────
def esearch(query, email, api_key, retmax=15):
    params = {
        "db": "pubmed", "term": query, "retmax": retmax,
        "retmode": "json", "email": email, "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key
    try:
        r = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        return []


def efetch(pmids, email, api_key):
    if not pmids:
        return []
    params = {
        "db": "pubmed", "id": ",".join(pmids),
        "retmode": "xml", "email": email,
    }
    if api_key:
        params["api_key"] = api_key
    try:
        r = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=30)
        r.raise_for_status()
    except Exception as e:
        return []

    papers = []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return []

    for article in root.findall(".//PubmedArticle"):
        pmid = _xt(article, ".//PMID")
        if not pmid:
            continue

        title = _xt(article, ".//ArticleTitle")

        abstract_parts = article.findall(".//AbstractText")
        pieces = []
        for ap in abstract_parts:
            label = ap.get("Label", "")
            text = ap.text or ""
            pieces.append(f"{label}: {text}" if label else text)
        abstract = " ".join(pieces)

        je = article.find(".//Journal")
        jname = _xt(je, "Title") if je is not None else ""
        jissn = _xt(je, "ISSN") if je is not None else ""

        pde = article.find(".//PubDate")
        yr = _xt(pde, "Year") if pde is not None else ""
        mo = _xt(pde, "Month") if pde is not None else ""
        dy = _xt(pde, "Day") if pde is not None else ""
        if not yr and pde is not None:
            ml = _xt(pde, "MedlineDate")
            m = re.match(r'(\d{4})', ml) if ml else None
            if m:
                yr = m.group(1)
        pub_date = yr
        if yr and mo:
            pub_date += f"-{mo}"
            if dy:
                pub_date += f"-{dy}"

        doi = ""
        for aid in article.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").strip()

        ptypes = [pt.text.strip() for pt in article.findall(".//PublicationType") if pt.text]

        papers.append({
            "PaperID":          pmid,
            "Title":            title[:500],
            "DOI":              doi,
            "PublicationDate":  pub_date,
            "JournalName":      jname,
            "JournalISSN":      jissn,
            "PublicationTypes": ";".join(ptypes),
            "Abstract":         abstract[:4000],
        })
    return papers


def _xt(parent, tag):
    if parent is None:
        return ""
    el = parent.find(tag)
    return (el.text or "").strip() if el is not None and el.text else ""


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run(args):
    t0 = time.time()
    data_dir = args.data_dir
    output_papers = args.output or os.path.join(data_dir, "papers.csv")
    output_sources = os.path.join(os.path.dirname(output_papers), "paper_sources.csv")
    email = args.email
    api_key = args.api_key
    sleep_time = 0.11 if api_key else 0.34
    retmax = args.retmax

    print("=" * 65)
    print("  ProtEvidenceDB — PubMed Extraction v2 (Smart Queries)")
    print("=" * 65)
    print(f"  Data dir:       {data_dir}")
    print(f"  Output papers:  {output_papers}")
    print(f"  Output sources: {output_sources}")
    print(f"  API key:        {'yes (10 req/s)' if api_key else 'no (3 req/s)'}")
    print(f"  Sleep:          {sleep_time}s")
    print(f"  Retmax:         {retmax}")
    print()

    # ── Load ──
    data = load_all_data(data_dir)

    # ── Build queries ──
    print("\nBuilding tiered queries …")
    queries, counts = build_queries(
        data,
        max_t1=args.max_t1,
        max_t2=args.max_t2,
        max_t3=args.max_t3,
        max_t4=args.max_t4,
        max_t5=args.max_t5,
    )

    print(f"  T1 (gene + drug):                    {counts['T1']:>6}")
    print(f"  T2 (gene+syns+protname + drug):      {counts['T2']:>6}")
    print(f"  T3 (gene + mechanism + cancer):      {counts['T3']:>6}")
    print(f"  T4 (organ + gene + drug):            {counts['T4']:>6}")
    print(f"  T5 (organ + gene + therapy):         {counts['T5']:>6}")
    print(f"  Total:                               {len(queries):>6}")

    # ── Load existing (append mode) ──
    seen_pmids = set()
    existing_papers = []
    if args.append and os.path.exists(output_papers):
        existing_papers = load_csv(output_papers)
        seen_pmids = {p["PaperID"] for p in existing_papers}
        print(f"\n  Append mode: {len(seen_pmids)} existing papers")

    # ── Execute ──
    print(f"\nRunning {len(queries)} queries …\n")

    all_papers = list(existing_papers)
    source_rows = []
    new_count = 0
    tier_hits = defaultdict(int)
    tier_papers = defaultdict(int)

    for qi, (tier, query, gene, drug_id, drug_name, bm_id) in enumerate(queries):
        pmids = esearch(query, email, api_key, retmax=retmax)
        new_pmids = [p for p in pmids if p not in seen_pmids]

        if not new_pmids:
            time.sleep(sleep_time)
            if (qi + 1) % 100 == 0:
                _progress(qi + 1, len(queries), new_count, tier_hits, tier_papers, t0)
            continue

        time.sleep(sleep_time)
        fetched = efetch(new_pmids, email, api_key)

        if fetched:
            tier_hits[tier] += 1
            for paper in fetched:
                pid = paper["PaperID"]
                if pid not in seen_pmids:
                    seen_pmids.add(pid)
                    all_papers.append(paper)
                    new_count += 1
                    tier_papers[tier] += 1

                # Track source — even if paper already existed, log the link
                source_rows.append({
                    "PaperID":    pid,
                    "BiomarkerID": bm_id,
                    "DrugID":     drug_id,
                    "GeneSymbol": gene,
                    "DrugName":   drug_name,
                    "QueryTier":  tier,
                    "QueryString": query[:300],
                })

        time.sleep(sleep_time)

        if (qi + 1) % 100 == 0:
            _progress(qi + 1, len(queries), new_count, tier_hits, tier_papers, t0)

    # ── Write papers ──
    os.makedirs(os.path.dirname(output_papers) or ".", exist_ok=True)
    with open(output_papers, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PAPER_FIELDS)
        w.writeheader()
        for p in all_papers:
            w.writerow(p)

    # ── Write paper_sources (the biomarker↔drug↔paper mapping) ──
    with open(output_sources, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SOURCE_FIELDS)
        w.writeheader()
        for s in source_rows:
            w.writerow(s)

    elapsed = time.time() - t0

    # ── Summary ──
    print(f"\n{'=' * 65}")
    print(f"  COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Queries executed:    {len(queries)}")
    print(f"  New papers:          {new_count}")
    print(f"  Total papers:        {len(all_papers)}")
    print(f"  Source links:        {len(source_rows)}  (paper ↔ biomarker ↔ drug)")
    print(f"  Elapsed:             {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print()
    print(f"  Papers per tier:")
    for tier in ["T1", "T2", "T3", "T4", "T5"]:
        h = tier_hits.get(tier, 0)
        p = tier_papers.get(tier, 0)
        print(f"    {tier}: {p:>6} papers from {h:>4} hit queries")
    print()
    print(f"  → {output_papers}  ({os.path.getsize(output_papers)/1024:.1f} KB)")
    print(f"  → {output_sources}  ({os.path.getsize(output_sources)/1024:.1f} KB)")
    print()
    print(f"  paper_sources.csv maps every paper to the (BiomarkerID, DrugID)")
    print(f"  pair that found it → use this to populate stances.csv")
    print(f"{'=' * 65}")


def _progress(done, total, new_count, tier_hits, tier_papers, t0):
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    tier_str = " | ".join([f"{t}:{tier_papers.get(t,0)}" for t in ["T1","T2","T3","T4","T5"]])
    print(f"  [{done:>5}/{total}]  new: {new_count:>5}  |  {tier_str}  |  "
          f"{elapsed:.0f}s  ETA {eta:.0f}s")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="ProtEvidenceDB — PubMed Extraction v2 (Smart Queries)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Query Tiers:
  T1  gene + drug                           (most precise, always run)
  T2  (gene OR synonyms OR protname) + drug (catches alias usage)
  T3  gene + mechanism + cancer             (catches papers using drug class)
  T4  organ + gene + drug                   (organ-contextualized)
  T5  organ + gene + generic therapy        (broadest, one per gene)

Examples:
  # Full run with API key
  python3 extract_pubmed.py --api-key KEY --max-t1 99999 --max-t2 99999

  # Quick test
  python3 extract_pubmed.py --max-t1 100 --max-t2 50 --max-t3 0 --max-t4 0 --max-t5 0

  # Append more T2/T3 results to existing papers
  python3 extract_pubmed.py --append --max-t1 0 --max-t2 99999 --max-t3 5000
        """,
    )
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--output", default=None,
                   help="Output path for papers.csv (default: DATA_DIR/papers.csv)")
    p.add_argument("--email", default="student@university.edu")
    p.add_argument("--api-key", default=None,
                   help="NCBI API key (10 req/s; get at ncbi.nlm.nih.gov/account/settings)")
    p.add_argument("--retmax", type=int, default=15,
                   help="Max papers per query (default: 15)")
    p.add_argument("--append", action="store_true",
                   help="Append to existing papers.csv")

    p.add_argument("--max-t1", type=int, default=99999,
                   help="Max T1 queries: gene+drug (default: unlimited)")
    p.add_argument("--max-t2", type=int, default=99999,
                   help="Max T2 queries: gene+synonyms+drug (default: unlimited)")
    p.add_argument("--max-t3", type=int, default=2000,
                   help="Max T3 queries: gene+mechanism+cancer (default: 2000)")
    p.add_argument("--max-t4", type=int, default=0,
                   help="Max T4 queries: organ+gene+drug (default: 0, redundant with T1)")
    p.add_argument("--max-t5", type=int, default=500,
                   help="Max T5 queries: organ+gene+therapy (default: 500)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)