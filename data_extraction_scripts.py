#!/usr/bin/env python3
"""
ProtEvidenceDB — Complete Data Extraction Pipeline
====================================================
Extracts all data needed for the ProtEvidenceDB relational database.

Sources:
  1. EDRN           → biomarkers.csv
  2. UniProt         → uniprot_proteins.csv, biomarker_uniprot_map.csv
  3. DGIdb (GraphQL) → drugs.csv, drug_gene_interactions.csv
  4. OpenAlex        → journals.csv  (SCImago fallback if CSV provided)
  5. Reactome / KEGG → pathways.csv, drug_pathways.csv  (derived from UniProt xrefs)
  6. Schema stubs    → stances.csv, users.csv, user_annotations.csv

PubMed extraction is in a separate script (extract_pubmed.py) because
it takes 20–30 min and benefits from an NCBI API key.

Requirements:
    pip install requests

Usage:
    python3 extract_pipeline.py                        # defaults
    python3 extract_pipeline.py --output-dir ./data    # custom output
    python3 extract_pipeline.py --skip edrn uniprot    # skip steps
    python3 extract_pipeline.py --scimago-csv /path/to/scimagojr.csv

SCImago note:
    scimagojr.com is behind Cloudflare and cannot be scraped.
    We use OpenAlex (open API) as the primary journal source.
    If you manually download the SCImago CSV from their website:
      1. Go to https://www.scimagojr.com/journalrank.php?category=2730
         (Oncology) or category=1306 (Cancer Research)
      2. Click "Download data" → saves a CSV
      3. Pass it with --scimago-csv /path/to/file.csv
    The script will parse it and merge with OpenAlex data.
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

# ─────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "./output_csvs"
HEADERS = {"User-Agent": "ProtEvidenceDB/1.0 (academic project; cancer biomarker database)"}

EDRN_URL       = "https://edrn.cancer.gov/data-and-resources/biomarkers/?ajax=json"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
DGIDB_GRAPHQL  = "https://dgidb.org/api/graphql"
OPENALEX_URL   = "https://api.openalex.org/sources"


# ═════════════════════════════════════════════
#  1. EDRN BIOMARKERS
# ═════════════════════════════════════════════
def extract_edrn(output_dir):
    """
    Pull all biomarkers from EDRN and filter to:
      Gene, Protein, Proteomic, Genomic, Genetic types.

    Output: biomarkers.csv
    Fields: BiomarkerID, Name, Type, Organ, Phase, Description,
            Assay, GeneSymbol, ExternalID, AlphaFoldID
    """
    print_header("1. EDRN Biomarkers")

    r = requests.get(EDRN_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    all_bm = r.json()["data"]
    print(f"  Fetched {len(all_bm)} total biomarkers from EDRN")

    WANTED = {"Gene", "Protein", "Genomic", "Genetic", "Proteomic"}
    TYPE_MAP = {
        "Protein":   "proteomic",
        "Proteomic": "proteomic",
        "Gene":      "genomic",
        "Genomic":   "genomic",
        "Genetic":   "genomic",
    }

    rows = []
    bid = 1
    for b in all_bm:
        kind = (b.get("kind") or "").strip()
        if kind not in WANTED:
            continue

        btype = TYPE_MAP.get(kind, kind.lower())
        edrn_id = b.get("identifier", "")

        # Phase: take highest if multi-valued
        phase_raw = str(b.get("phases", ""))
        phases = [p.strip() for p in phase_raw.replace("|", ",").split(",")
                  if p.strip().isdigit()]
        phase = max(phases, key=int) if phases else ""

        # Gene symbol heuristic
        title = b.get("title", "").strip()
        gene_symbol = ""
        if kind in ("Gene", "Genetic", "Genomic"):
            if re.match(r'^[A-Z0-9][A-Z0-9\-]{0,14}$', title):
                gene_symbol = title
        elif kind in ("Protein", "Proteomic"):
            slug = b.get("url", "").rstrip("/").split("/")[-1].upper()
            if re.match(r'^[A-Z0-9][A-Z0-9\-]{0,14}$', slug):
                gene_symbol = slug

        rows.append({
            "BiomarkerID":  f"BM{bid:05d}",
            "Name":         title,
            "Type":         btype,
            "Organ":        b.get("organs", ""),
            "Phase":        phase,
            "Description":  (b.get("description", "") or "")[:2000],
            "Assay":        "",
            "GeneSymbol":   gene_symbol,
            "ExternalID":   edrn_id,
            "AlphaFoldID":  "",
        })
        bid += 1

    path = write_csv(output_dir, "biomarkers.csv", rows)
    print(f"  → {len(rows)} biomarkers → {path}")

    # Stats
    types = defaultdict(int)
    for r in rows:
        types[r["Type"]] += 1
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"      {t}: {c}")

    return rows


# ═════════════════════════════════════════════
#  2. UNIPROT PROTEINS (batched per-biomarker)
# ═════════════════════════════════════════════
UNIPROT_FIELDS = ",".join([
    "accession", "id", "gene_names", "protein_name", "organism_name",
    "length", "mass",
    "cc_function", "cc_disease", "cc_subcellular_location",
    "cc_tissue_specificity", "cc_pathway",
    "keyword",
    "ft_domain", "ft_binding", "ft_act_site",
    "xref_alphafolddb", "xref_pdb", "xref_drugbank", "xref_chembl",
    "xref_kegg", "xref_reactome",
    "reviewed",
])


def extract_uniprot(output_dir, biomarkers):
    """
    For each biomarker with a GeneSymbol, query UniProt (batched 20 genes
    per request) and extract all protein fields.

    Fallback: for proteomic biomarkers without gene symbols, search by name.

    Output:
      - uniprot_proteins.csv       (one row per unique UniProt accession)
      - biomarker_uniprot_map.csv  (BiomarkerID → UniProtAccession)
      - updates AlphaFoldID in biomarkers list (returned)
    """
    print_header("2. UniProt Proteins")

    # Collect unique gene symbols and no-gene biomarkers
    gene_to_bms = defaultdict(list)
    no_gene_proteomic = []

    for bm in biomarkers:
        gene = bm.get("GeneSymbol", "").strip()
        if gene:
            gene_to_bms[gene].append(bm)
        elif bm.get("Type") == "proteomic":
            no_gene_proteomic.append(bm)

    unique_genes = sorted(gene_to_bms.keys())
    print(f"  Unique gene symbols: {len(unique_genes)}")
    print(f"  Proteomic without gene (name fallback): {len(no_gene_proteomic)}")

    # ── Phase 1: batch gene queries ──
    print(f"\n  Phase 1: Batch gene queries (20 per request) …")
    gene_results = {}   # gene → parsed entry
    BATCH = 20

    for i in range(0, len(unique_genes), BATCH):
        batch = unique_genes[i:i + BATCH]
        result_map = _uniprot_batch_genes(batch)
        for gene, entry in result_map.items():
            gene_results[gene] = _parse_uniprot(entry)
        time.sleep(0.3)

        if (i // BATCH) % 5 == 0 and i > 0:
            print(f"    {min(i + BATCH, len(unique_genes))}/{len(unique_genes)} genes → "
                  f"{len(gene_results)} matched")

    print(f"    Gene phase: {len(gene_results)}/{len(unique_genes)} matched")

    # ── Phase 2: name fallback ──
    print(f"\n  Phase 2: Name-based lookup for {len(no_gene_proteomic)} proteomic biomarkers …")
    name_results = {}
    name_hits = 0
    for bm in no_gene_proteomic:
        name = bm.get("Name", "").strip()
        if not name or name in name_results or len(name) < 4:
            continue
        entry = _uniprot_search_name(name)
        if entry:
            name_results[name] = _parse_uniprot(entry)
            name_hits += 1
        time.sleep(0.2)
    print(f"    Name phase: {name_hits} matches")

    # ── Build outputs ──
    proteins = []
    bm_map = []
    seen_acc = set()

    for bm in biomarkers:
        gene = bm.get("GeneSymbol", "").strip()
        name = bm.get("Name", "").strip()
        parsed = None
        method = ""

        if gene and gene in gene_results:
            parsed = gene_results[gene]
            method = "gene_symbol"
        elif name in name_results:
            parsed = name_results[name]
            method = "name_search"

        if parsed:
            acc = parsed["UniProtAccession"]
            bm["AlphaFoldID"] = parsed["AlphaFoldID"]

            if acc not in seen_acc:
                seen_acc.add(acc)
                proteins.append(parsed)

            bm_map.append({
                "BiomarkerID":      bm["BiomarkerID"],
                "BiomarkerName":    name,
                "UniProtAccession": acc,
                "UniProtGene":      parsed["GeneSymbol"],
                "ProteinName":      parsed["ProteinName"],
                "MatchMethod":      method,
            })

    # Write
    path1 = write_csv(output_dir, "uniprot_proteins.csv", proteins)
    path2 = write_csv(output_dir, "biomarker_uniprot_map.csv", bm_map)

    # Overwrite biomarkers.csv with AlphaFoldIDs filled in
    write_csv(output_dir, "biomarkers.csv", biomarkers)

    print(f"\n  → {len(proteins)} unique proteins → {path1}")
    print(f"  → {len(bm_map)} biomarker↔protein mappings → {path2}")
    print(f"  → biomarkers.csv updated with AlphaFoldIDs")

    # Fill rate
    filled = sum(1 for p in proteins if p.get("Function"))
    print(f"  Fill rates: Function={filled}/{len(proteins)}, "
          f"AlphaFold={sum(1 for p in proteins if p.get('AlphaFoldID'))}/{len(proteins)}, "
          f"Reactome={sum(1 for p in proteins if p.get('ReactomePathways'))}/{len(proteins)}")

    return proteins, bm_map


def _uniprot_batch_genes(gene_list):
    """Query UniProt for a batch of gene symbols. Return {gene: raw_entry}."""
    gene_or = " OR ".join([f'gene_exact:"{g}"' for g in gene_list])
    q = f"(organism_id:9606) AND ({gene_or})"

    all_results = []
    cursor = None

    for _ in range(5):
        params = {"query": q, "format": "json", "size": 100, "fields": UNIPROT_FIELDS}
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get(UNIPROT_SEARCH, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            all_results.extend(results)
            link = r.headers.get("Link", "")
            cursor = None
            if 'rel="next"' in link:
                m = re.search(r'cursor=([^&>]+)', link)
                if m:
                    cursor = m.group(1)
            if not cursor:
                break
        except Exception as e:
            print(f"    Batch query error: {e}")
            break

    # Map gene → best entry (prefer reviewed)
    gene_map = {}
    for entry in all_results:
        genes_data = entry.get("genes", [])
        if not genes_data:
            continue
        primary = genes_data[0].get("geneName", {}).get("value", "")
        syns = [s.get("value", "") for s in genes_data[0].get("synonyms", [])]
        all_names = [primary] + syns
        is_rev = "reviewed" in entry.get("entryType", "").lower()

        for gn in all_names:
            gu = gn.upper()
            for orig in gene_list:
                if orig.upper() == gu:
                    if orig not in gene_map:
                        gene_map[orig] = entry
                    elif is_rev and "reviewed" not in gene_map[orig].get("entryType", "").lower():
                        gene_map[orig] = entry
    return gene_map


def _uniprot_search_name(name):
    """Fallback: search UniProt by protein name, human only."""
    clean = re.sub(r'\s*\(.*?\)\s*', ' ', name).strip()
    clean = re.sub(r'\s+', ' ', clean)
    if len(clean) < 3 or len(clean) > 80:
        return None
    q = f'(protein_name:"{clean}") AND (organism_id:9606)'
    try:
        r = requests.get(UNIPROT_SEARCH, params={
            "query": q, "format": "json", "size": 3, "fields": UNIPROT_FIELDS,
        }, headers=HEADERS, timeout=20)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        reviewed = [e for e in results if "reviewed" in e.get("entryType", "").lower()]
        return reviewed[0] if reviewed else results[0]
    except:
        return None


def _parse_uniprot(entry):
    """Parse a single UniProt JSON entry into a flat dict with all fields."""
    acc = entry.get("primaryAccession", "")

    # Gene names
    genes_data = entry.get("genes", [])
    gene_primary = ""
    gene_synonyms = []
    if genes_data:
        gn = genes_data[0]
        gene_primary = gn.get("geneName", {}).get("value", "")
        gene_synonyms = [s.get("value", "") for s in gn.get("synonyms", [])]

    # Protein name
    pd = entry.get("proteinDescription", {})
    prot_name = ""
    for key in ["recommendedName", "submissionNames", "alternativeNames"]:
        src = pd.get(key)
        if src:
            if isinstance(src, list):
                src = src[0] if src else {}
            prot_name = src.get("fullName", {}).get("value", "")
            if prot_name:
                break

    # Comments
    comments = entry.get("comments", [])
    function_text = ""
    disease_list = []
    subcellular = ""
    tissue_spec = ""
    pathway_text = ""

    for c in comments:
        ct = c.get("commentType", "")
        if ct == "FUNCTION":
            txts = c.get("texts", [])
            if txts:
                function_text = txts[0].get("value", "")
        elif ct == "DISEASE":
            d = c.get("disease", {})
            if d:
                did = d.get("diseaseId", "")
                dacr = d.get("acronym", "")
                disease_list.append(f"{did}" + (f" ({dacr})" if dacr else ""))
        elif ct == "SUBCELLULAR LOCATION":
            locs = c.get("subcellularLocations", [])
            subcellular = "; ".join([
                loc.get("location", {}).get("value", "")
                for loc in locs if loc.get("location")
            ])
        elif ct == "TISSUE SPECIFICITY":
            txts = c.get("texts", [])
            if txts:
                tissue_spec = txts[0].get("value", "")
        elif ct == "PATHWAY":
            txts = c.get("texts", [])
            if txts:
                pathway_text = txts[0].get("value", "")

    # Keywords
    keywords = [kw.get("name", "") for kw in entry.get("keywords", [])]

    # Features
    features = entry.get("features", [])
    domains, binding_sites, active_sites = [], [], []
    for ft in features:
        ft_type = ft.get("type", "")
        desc = ft.get("description", "") or ""
        loc = ft.get("location", {})
        s = loc.get("start", {}).get("value", "")
        e = loc.get("end", {}).get("value", "")
        pos = f"{s}-{e}" if s and e else ""
        label = f"{desc} [{pos}]" if pos else desc
        if ft_type == "Domain":
            domains.append(label)
        elif ft_type == "Binding site":
            binding_sites.append(label)
        elif ft_type == "Active site":
            active_sites.append(label)

    # Cross-references
    xrefs = entry.get("uniProtKBCrossReferences", [])
    alphafold_id = ""
    pdb_ids, drugbank_ids, chembl_ids, kegg_ids = [], [], [], []
    reactome_entries = []

    for xr in xrefs:
        db = xr.get("database", "")
        xid = xr.get("id", "")
        props = {p.get("key", ""): p.get("value", "") for p in xr.get("properties", [])}
        if db == "AlphaFoldDB":
            alphafold_id = xid
        elif db == "PDB":
            pdb_ids.append(xid)
        elif db == "DrugBank":
            drugbank_ids.append(xid)
        elif db == "ChEMBL":
            chembl_ids.append(xid)
        elif db == "KEGG":
            kegg_ids.append(xid)
        elif db == "Reactome":
            rn = props.get("PathwayName", "")
            reactome_entries.append(f"{xid}:{rn}" if rn else xid)

    seq = entry.get("sequence", {})

    return {
        "UniProtAccession":     acc,
        "UniProtID":            entry.get("uniProtkbId", ""),
        "EntryType":            "reviewed" if "reviewed" in entry.get("entryType", "").lower() else "unreviewed",
        "GeneSymbol":           gene_primary,
        "GeneSynonyms":         "|".join(gene_synonyms),
        "ProteinName":          prot_name,
        "Function":             function_text[:3000],
        "DiseaseAssociations":  "|".join(disease_list),
        "SubcellularLocation":  subcellular[:500],
        "TissueSpecificity":    tissue_spec[:1000],
        "PathwayAnnotation":    pathway_text[:500],
        "Keywords":             "|".join(keywords),
        "Domains":              "|".join(domains[:20]),
        "BindingSites":         "|".join(binding_sites[:20]),
        "ActiveSites":          "|".join(active_sites[:10]),
        "AlphaFoldID":          alphafold_id,
        "PDB_IDs":              "|".join(pdb_ids[:30]),
        "DrugBankIDs":          "|".join(drugbank_ids[:15]),
        "ChEMBL_IDs":           "|".join(chembl_ids[:15]),
        "KEGG_IDs":             "|".join(kegg_ids[:10]),
        "ReactomePathways":     "|".join(reactome_entries[:20]),
        "SequenceLength":       seq.get("length", ""),
        "MolecularWeight":      seq.get("molWeight", ""),
        "Organism":             "Homo sapiens",
    }


# ═════════════════════════════════════════════
#  3. DGIdb DRUGS + INTERACTIONS (GraphQL)
# ═════════════════════════════════════════════
def extract_dgidb(output_dir, biomarkers):
    """
    Query DGIdb GraphQL API for drug–gene interactions.
    Batches 25 gene symbols per request.

    Output:
      - drugs.csv                  (unique drugs)
      - drug_gene_interactions.csv (all interaction records)
    """
    print_header("3. DGIdb Drug–Gene Interactions")

    gene_symbols = sorted({bm["GeneSymbol"] for bm in biomarkers if bm.get("GeneSymbol")})
    print(f"  Querying DGIdb for {len(gene_symbols)} gene symbols …")

    drugs_rows = []
    interactions_rows = []
    seen_drugs = {}
    did = 1
    iid = 1
    BATCH = 25

    for i in range(0, len(gene_symbols), BATCH):
        batch = gene_symbols[i:i + BATCH]
        genes_json = json.dumps(batch)

        query = f"""{{
          genes(names: {genes_json}) {{
            nodes {{
              name
              longName
              interactions {{
                drug {{
                  name
                  conceptId
                  approved
                  drugAttributes {{ name value }}
                }}
                interactionScore
                interactionTypes {{ type directionality }}
                publications {{ pmid }}
                sources {{ sourceDbName }}
              }}
            }}
          }}
        }}"""

        try:
            r = requests.post(
                DGIDB_GRAPHQL,
                json={"query": query},
                headers={**HEADERS, "Content-Type": "application/json"},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    Batch {i} error: {e}")
            time.sleep(2)
            continue

        if "errors" in data:
            print(f"    GraphQL errors: {data['errors'][:1]}")
            continue

        for gene_node in data.get("data", {}).get("genes", {}).get("nodes", []):
            gene_name = gene_node.get("name", "")
            for ix in gene_node.get("interactions", []):
                drug_data = ix.get("drug", {})
                drug_name = drug_data.get("name", "")
                concept_id = drug_data.get("conceptId", "")
                approved = drug_data.get("approved", False)

                attrs = {}
                for a in drug_data.get("drugAttributes", []):
                    attrs[a.get("name", "")] = a.get("value", "")

                ix_types = ix.get("interactionTypes", [])
                mechanism = "; ".join([t.get("type", "") for t in ix_types if t.get("type")])
                directionality = "; ".join([t.get("directionality", "") for t in ix_types if t.get("directionality")])

                pubs = [str(p.get("pmid", "")) for p in ix.get("publications", []) if p.get("pmid")]
                sources = [s.get("sourceDbName", "") for s in ix.get("sources", []) if s.get("sourceDbName")]

                # Drug dedup
                drug_key = concept_id or drug_name
                if drug_key not in seen_drugs:
                    seen_drugs[drug_key] = f"D{did:05d}"
                    drugs_rows.append({
                        "DrugID":            seen_drugs[drug_key],
                        "DrugDBID":          concept_id,
                        "Name":              drug_name,
                        "Modality":          attrs.get("Drug Class", ""),
                        "MechanismOfAction": mechanism,
                        "MaxTrialPhase":     attrs.get("Clinical Trial Phase", ""),
                        "IsApproved":        approved,
                        "HasBeenWithdrawn":  "",
                        "SMILES":            attrs.get("SMILES", ""),
                        "MolecularWeight":   attrs.get("Molecular Weight", ""),
                    })
                    did += 1

                interactions_rows.append({
                    "InteractionID":   f"IX{iid:05d}",
                    "GeneSymbol":      gene_name,
                    "DrugID":          seen_drugs[drug_key],
                    "DrugName":        drug_name,
                    "ConceptID":       concept_id,
                    "InteractionType": mechanism,
                    "Directionality":  directionality,
                    "Score":           ix.get("interactionScore", ""),
                    "PMIDs":           "|".join(pubs),
                    "Sources":         "|".join(sources),
                })
                iid += 1

        time.sleep(0.5)

        if (i // BATCH) % 5 == 0 and i > 0:
            print(f"    {min(i + BATCH, len(gene_symbols))}/{len(gene_symbols)} genes → "
                  f"{len(drugs_rows)} drugs, {len(interactions_rows)} interactions")

    path_d = write_csv(output_dir, "drugs.csv", drugs_rows)
    path_i = write_csv(output_dir, "drug_gene_interactions.csv", interactions_rows)

    print(f"\n  → {len(drugs_rows)} drugs → {path_d}")
    print(f"  → {len(interactions_rows)} interactions → {path_i}")
    print(f"  Approved drugs: {sum(1 for d in drugs_rows if d['IsApproved'])}")

    return drugs_rows, interactions_rows


# ═════════════════════════════════════════════
#  4. JOURNALS (OpenAlex + optional SCImago)
# ═════════════════════════════════════════════
def extract_journals(output_dir, scimago_csv=None):
    """
    Retrieve cancer/oncology journal rankings.

    Primary: OpenAlex API (open, no auth, no Cloudflare).
    Optional: if --scimago-csv is provided, parse and merge it.

    Output: journals.csv
    Fields: JournalID, Name, ISSN, SCI_Ranking, ImpactFactor, Publisher
    """
    print_header("4. Journal Rankings")

    journals = []
    seen_issn = set()
    jid = 1

    # ── OpenAlex ──
    print("  Querying OpenAlex for cancer/oncology journals …")
    search_terms = [
        "oncology", "cancer research", "cancer biology",
        "clinical oncology", "neoplasm", "tumor",
        "carcinogenesis", "leukemia", "breast cancer",
        "lung cancer", "prostate cancer", "immunotherapy cancer",
        "molecular cancer", "cancer genetics", "surgical oncology",
        "radiation oncology", "cancer prevention", "cancer epidemiology",
        "pediatric oncology", "cancer therapy",
    ]

    for term in search_terms:
        try:
            r = requests.get(OPENALEX_URL, params={
                "search": term, "per_page": 50, "filter": "type:journal",
            }, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    OpenAlex query '{term}' failed: {e}")
            continue

        for src in data.get("results", []):
            issn_l = src.get("issn_l", "")
            if not issn_l or issn_l in seen_issn:
                continue
            seen_issn.add(issn_l)

            stats = src.get("summary_stats", {})
            h_index = stats.get("h_index", "")
            citedness = stats.get("2yr_mean_citedness", "")

            journals.append({
                "JournalID":    f"J{jid:05d}",
                "Name":         src.get("display_name", ""),
                "ISSN":         issn_l,
                "SCI_Ranking":  h_index,
                "ImpactFactor": round(citedness, 3) if isinstance(citedness, (int, float)) else "",
                "Publisher":    src.get("host_organization_name", ""),
            })
            jid += 1

        time.sleep(0.3)

    print(f"    OpenAlex: {len(journals)} journals")

    # ── SCImago CSV (optional) ──
    if scimago_csv and os.path.exists(scimago_csv):
        print(f"\n  Parsing SCImago CSV: {scimago_csv}")
        scimago_count = 0

        try:
            # SCImago CSVs use semicolons as delimiters
            with open(scimago_csv, encoding="utf-8") as f:
                # Try to detect delimiter
                first_line = f.readline()
                f.seek(0)
                delimiter = ";" if ";" in first_line else ","

                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    # SCImago columns: Rank;Sourceid;Title;Type;Issn;SJR;...
                    title = row.get("Title", "").strip()
                    issn_raw = row.get("Issn", row.get("ISSN", "")).strip()
                    sjr = row.get("SJR", "").strip()
                    publisher = row.get("Publisher", "").strip()
                    hindex = row.get("H index", row.get("H Index", "")).strip()

                    # Parse ISSN (SCImago format: "00085472, 15387445")
                    issns = [i.strip() for i in issn_raw.replace('"', '').split(",")]
                    issn = issns[0] if issns else ""
                    # Format as XXXX-XXXX
                    if len(issn) == 8 and issn.isalnum():
                        issn = f"{issn[:4]}-{issn[4:]}"

                    if not title or issn in seen_issn:
                        continue
                    seen_issn.add(issn)

                    journals.append({
                        "JournalID":    f"J{jid:05d}",
                        "Name":         title,
                        "ISSN":         issn,
                        "SCI_Ranking":  hindex or sjr,
                        "ImpactFactor": sjr,
                        "Publisher":    publisher,
                    })
                    jid += 1
                    scimago_count += 1

            print(f"    SCImago: added {scimago_count} new journals")
        except Exception as e:
            print(f"    SCImago parse error: {e}")

    path = write_csv(output_dir, "journals.csv", journals)
    print(f"\n  → {len(journals)} journals → {path}")
    return journals


# ═════════════════════════════════════════════
#  5. PATHWAYS + DRUG-PATHWAY (from UniProt xrefs)
# ═════════════════════════════════════════════
def build_pathways(output_dir, proteins, interactions):
    """
    Build the pathway and drug-pathway tables using REAL biological pathways
    from UniProt cross-references (Reactome + KEGG).

    Chain: Drug → (interaction) → Gene → (uniprot) → Reactome/KEGG pathways

    Output:
      - pathways.csv       (unique Reactome + KEGG pathway records)
      - drug_pathways.csv  (DrugID ↔ PathwayID junction)
    """
    print_header("5. Pathways (Reactome / KEGG)")

    # gene → {reactome: [(id, name)], kegg: [id]}
    gene_pathways = {}
    for prot in proteins:
        gene = prot.get("GeneSymbol", "")
        if not gene:
            continue

        reactome = []
        raw = prot.get("ReactomePathways", "").strip()
        if raw:
            for entry in raw.split("|"):
                entry = entry.strip()
                m = re.match(r'(R-HSA-\d+):(.*)', entry)
                if m:
                    reactome.append((m.group(1), m.group(2).strip()))
                elif entry:
                    reactome.append((entry, ""))

        kegg = []
        raw_k = prot.get("KEGG_IDs", "").strip()
        if raw_k:
            kegg = [k.strip() for k in raw_k.split("|") if k.strip()]

        gene_pathways[gene] = {"reactome": reactome, "kegg": kegg}

    # drug → genes (from interactions)
    drug_genes = defaultdict(set)
    drug_names = {}
    for ix in interactions:
        did = ix["DrugID"]
        gene = ix["GeneSymbol"].strip()
        drug_genes[did].add(gene)
        if did not in drug_names:
            drug_names[did] = ix.get("DrugName", "")

    print(f"  Genes with pathways: {len(gene_pathways)}")
    print(f"  Drugs with interactions: {len(drug_genes)}")

    # Build
    pw_registry = {}   # (db, pw_id) → PathwayID
    pathways_rows = []
    pw_counter = 1

    dp_rows_raw = []

    for drug_id, genes in drug_genes.items():
        for gene in genes:
            gpw = gene_pathways.get(gene)
            if not gpw:
                continue

            for (rid, rname) in gpw["reactome"]:
                key = ("Reactome", rid)
                if key not in pw_registry:
                    pid = f"PW{pw_counter:05d}"
                    pw_registry[key] = pid
                    pathways_rows.append({
                        "PathwayID":   pid,
                        "Name":        rname if rname else rid,
                        "PathwayDBID": rid,
                        "Description": f"Reactome pathway {rid}",
                    })
                    pw_counter += 1
                dp_rows_raw.append((drug_id, pw_registry[key], "Reactome"))

            for kid in gpw["kegg"]:
                key = ("KEGG", kid)
                if key not in pw_registry:
                    pid = f"PW{pw_counter:05d}"
                    pw_registry[key] = pid
                    pathways_rows.append({
                        "PathwayID":   pid,
                        "Name":        kid,
                        "PathwayDBID": kid,
                        "Description": f"KEGG gene entry {kid}",
                    })
                    pw_counter += 1
                dp_rows_raw.append((drug_id, pw_registry[key], "KEGG"))

    # Dedup drug-pathway
    seen_dp = set()
    dp_rows = []
    dp_id = 1
    for (did, pid, src) in dp_rows_raw:
        key = (did, pid)
        if key not in seen_dp:
            seen_dp.add(key)
            dp_rows.append({
                "DrugPathwayID":  f"DP{dp_id:05d}",
                "DrugID":         did,
                "PathwayID":      pid,
                "EvidenceSource": src,
            })
            dp_id += 1

    reactome_ct = sum(1 for p in pathways_rows if p["PathwayDBID"].startswith("R-HSA"))
    kegg_ct = len(pathways_rows) - reactome_ct

    path_pw = write_csv(output_dir, "pathways.csv", pathways_rows)
    path_dp = write_csv(output_dir, "drug_pathways.csv", dp_rows)

    print(f"\n  → {len(pathways_rows)} pathways ({reactome_ct} Reactome, {kegg_ct} KEGG) → {path_pw}")
    print(f"  → {len(dp_rows)} drug↔pathway links (deduped from {len(dp_rows_raw)}) → {path_dp}")

    return pathways_rows, dp_rows


# ═════════════════════════════════════════════
#  6. SCHEMA TABLES (stances, users, annotations)
# ═════════════════════════════════════════════
def build_schema_stubs(output_dir, biomarkers, drugs, interactions):
    """
    Build stub tables for:
      - stances.csv          (Drug↔Biomarker↔Paper links, default stance=unclear)
      - users.csv            (template with 3 seed users)
      - user_annotations.csv (empty template with headers)
    """
    print_header("6. Schema Tables (stances, users, annotations)")

    # Stances: for each interaction with PMIDs, create a stub
    bm_by_gene = {}
    for bm in biomarkers:
        gs = bm.get("GeneSymbol", "")
        if gs:
            bm_by_gene[gs] = bm["BiomarkerID"]

    stances = []
    sid = 1
    for ix in interactions:
        gene = ix.get("GeneSymbol", "")
        drug_id = ix.get("DrugID", "")
        pmids = (ix.get("PMIDs", "") or "").split("|")
        bm_id = bm_by_gene.get(gene, "")
        if not bm_id or not drug_id:
            continue
        for pmid in pmids:
            pmid = pmid.strip()
            if not pmid:
                continue
            stances.append({
                "StanceID":    f"ST{sid:05d}",
                "DrugID":      drug_id,
                "BiomarkerID": bm_id,
                "PaperID":     pmid,
                "JournalID":   "",
                "StanceValue": "unclear",
            })
            sid += 1

    # Users
    users = [
        {"UserID": "U00001", "UserName": "admin",      "Email": "admin@protevidencedb.org"},
        {"UserID": "U00002", "UserName": "annotator1",  "Email": "annotator1@protevidencedb.org"},
        {"UserID": "U00003", "UserName": "annotator2",  "Email": "annotator2@protevidencedb.org"},
    ]

    # Annotations (empty template)
    annotations_header = [
        "UserAnnotationID", "UserID", "PaperID",
        "BiomarkerID", "DrugID", "AnnotationText", "AnnotationType",
    ]

    write_csv(output_dir, "stances.csv", stances)
    write_csv(output_dir, "users.csv", users)

    # Write empty annotations with just headers
    ann_path = os.path.join(output_dir, "user_annotations.csv")
    with open(ann_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=annotations_header)
        w.writeheader()

    print(f"  → {len(stances)} stance stubs → stances.csv")
    print(f"  → {len(users)} seed users → users.csv")
    print(f"  → user_annotations.csv (empty template)")


# ═════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════
def write_csv(output_dir, filename, rows):
    """Write list of dicts to CSV. Returns full path."""
    path = os.path.join(output_dir, filename)
    if not rows:
        # Write empty file with no headers
        with open(path, "w", newline="", encoding="utf-8") as f:
            pass
        return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_header(title):
    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print(f"{'═' * 65}")


def print_summary(output_dir):
    """Print final file summary."""
    print(f"\n{'═' * 65}")
    print(f"  OUTPUT SUMMARY")
    print(f"{'═' * 65}")
    for fn in sorted(os.listdir(output_dir)):
        if not fn.endswith(".csv"):
            continue
        fp = os.path.join(output_dir, fn)
        with open(fp, encoding="utf-8") as f:
            rows = sum(1 for _ in csv.reader(f)) - 1
        size = os.path.getsize(fp) / 1024
        print(f"  {fn:42s}  {rows:>7,} rows  ({size:>8.1f} KB)")
    print(f"{'═' * 65}")


# ═════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="ProtEvidenceDB — Complete Data Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 extract_pipeline.py
  python3 extract_pipeline.py --output-dir ./my_data
  python3 extract_pipeline.py --scimago-csv ~/Downloads/scimagojr.csv
  python3 extract_pipeline.py --skip uniprot     # skip slow step
  python3 extract_pipeline.py --skip dgidb        # skip if already have drugs.csv
        """,
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--scimago-csv", default=None,
        help="Path to manually downloaded SCImago CSV (optional, merged with OpenAlex)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["edrn", "uniprot", "dgidb", "journals", "pathways", "stubs"],
        help="Skip specific extraction steps",
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    print("╔" + "═" * 63 + "╗")
    print("║  ProtEvidenceDB — Complete Data Extraction Pipeline           ║")
    print("╚" + "═" * 63 + "╝")
    print(f"  Output:  {os.path.abspath(output_dir)}")
    print(f"  Skip:    {args.skip or 'none'}")

    # ── Step 1: EDRN ──
    if "edrn" not in args.skip:
        biomarkers = extract_edrn(output_dir)
    else:
        print_header("1. EDRN — SKIPPED (loading from file)")
        biomarkers = load_existing_csv(output_dir, "biomarkers.csv")

    # ── Step 2: UniProt ──
    if "uniprot" not in args.skip:
        proteins, bm_map = extract_uniprot(output_dir, biomarkers)
    else:
        print_header("2. UniProt — SKIPPED (loading from file)")
        proteins = load_existing_csv(output_dir, "uniprot_proteins.csv")
        bm_map = load_existing_csv(output_dir, "biomarker_uniprot_map.csv")

    # ── Step 3: DGIdb ──
    if "dgidb" not in args.skip:
        drugs, interactions = extract_dgidb(output_dir, biomarkers)
    else:
        print_header("3. DGIdb — SKIPPED (loading from file)")
        drugs = load_existing_csv(output_dir, "drugs.csv")
        interactions = load_existing_csv(output_dir, "drug_gene_interactions.csv")

    # ── Step 4: Journals ──
    if "journals" not in args.skip:
        journals = extract_journals(output_dir, scimago_csv=args.scimago_csv)
    else:
        print_header("4. Journals — SKIPPED")

    # ── Step 5: Pathways ──
    if "pathways" not in args.skip:
        build_pathways(output_dir, proteins, interactions)
    else:
        print_header("5. Pathways — SKIPPED")

    # ── Step 6: Schema stubs ──
    if "stubs" not in args.skip:
        build_schema_stubs(output_dir, biomarkers, drugs, interactions)
    else:
        print_header("6. Schema stubs — SKIPPED")

    # ── Summary ──
    elapsed = time.time() - t0
    print_summary(output_dir)
    print(f"\n  Completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"\n  Next step: run extract_pubmed.py to fetch papers from PubMed.")
    print(f"    python3 extract_pubmed.py --data-dir {output_dir}\n")


def load_existing_csv(output_dir, filename):
    """Load an existing CSV for skipped steps."""
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — downstream steps may fail")
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"  Loaded {len(rows)} rows from {filename}")
    return rows


if __name__ == "__main__":
    main()