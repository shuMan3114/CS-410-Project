# CS-410 Recommender System

This document describes the website-oriented recommender in:

- `script/recommendation/content_based_recommender.py`

It is designed for repeated frontend/backend calls with persistent feedback state.

---

## 1) Design Goal

- Serve recommendations for both:
  - **Cold-start users** (no history, no query context)
  - **Personalized users** (with query filters and/or feedback history)
- Support iterative reranking loop after each feedback event.
- Keep stance generation separate from recommendation.

Stance input is only:

- `output_csvs/final_paper_stances.csv`

---

## 2) Core Inputs

- `data/raw/papers.csv`
- `data/raw/paper_sources.csv`
- `data/raw/biomarkers.csv`
- `data/raw/drug_gene_interactions.csv`
- `data/raw/biomarker_uniprot_map.csv`
- `output_csvs/final_paper_stances.csv`

Optional feedback inputs:

- `data/raw/user_annotations.csv`
- `output_csvs/recommender_feedback_state.json` (persistent state file)

---

## 3) Scoring Logic

### Personalized mode

Weighted additive score:

- organ match
- biomarker match
- drug match
- stance match
- text match
- interaction score
- structure score
- time score
- stance confidence contribution
- feedback delta (+/-)

### Cold-start mode

When no user profile/query/filter signal exists (or forced):

- confidence-based prior score
- recency score
- entity-density score
- plus interaction + structure support
- optional stance-diversified ranking

Response includes mode label:

- `mode = "cold_start"` or `mode = "personalized"`

---

## 4) Frontend Loop Contract

Each call can include:

- query filters
- weight overrides
- runtime feedback events

And each response returns:

- ranked recommendations
- score components per item
- mode metadata

Feedback loop behavior:

1. Frontend sends new `feedback_events`.
2. Backend calls recommender with same `--feedback-state-json`.
3. Script updates state and re-ranks.
4. Next request uses updated state automatically.

---

## 5) API Request/Response Files

Example request:

- `script/recommendation/recommender_api_request_example.json`

Cold-start example:

- `script/recommendation/recommender_api_request_cold_start_example.json`

Integration notes:

- `script/recommendation/recommender_frontend_integration.md`

---

## 6) Default Execution (API mode)

```bash
.venv/bin/python "script/recommendation/content_based_recommender.py" \
  --api-request-json "script/recommendation/recommender_api_request_example.json" \
  --api-response-json "output_csvs/recommender_api_response.json" \
  --disable-csv-output
```

Cold-start example:

```bash
.venv/bin/python "script/recommendation/content_based_recommender.py" \
  --api-request-json "script/recommendation/recommender_api_request_cold_start_example.json" \
  --api-response-json "output_csvs/recommender_api_response_cold_start.json" \
  --disable-csv-output
```

---

## 7) Practical Defaults

- stance source defaults to `output_csvs/final_paper_stances.csv`
- persistent feedback state defaults to `output_csvs/recommender_feedback_state.json`
- cold-start diversification is enabled by default
- all file paths default to project-local CSVs

This keeps deployment simple for a website backend wrapper.
