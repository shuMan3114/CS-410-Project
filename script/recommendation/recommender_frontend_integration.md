# Frontend Integration Guide

Use `content_based_recommender.py` as a request/response recommender service from your backend.

## Request Loop Pattern

Each frontend call should:

1. Send query + new feedback events.
2. Backend writes request payload JSON.
3. Backend runs recommender script.
4. Backend returns response JSON.
5. Repeat on next user action.

Feedback is accumulated per user through `--feedback-state-json`.

## CLI Command (Backend Side)

```bash
.venv/bin/python "script/recommendation/content_based_recommender.py" \
  --api-request-json "script/recommendation/recommender_api_request_example.json" \
  --api-response-json "output_csvs/recommender_api_response.json" \
  --feedback-state-json "output_csvs/recommender_feedback_state.json" \
  --disable-csv-output
```

## Required API Request Fields

- `user_id` (string)
- Optional `top_k` (int)
- Optional `query` object
  - `organs` (list[string])
  - `biomarkers` (list[string])
  - `drugs` (list[string])
  - `stances` (list[string])
  - `text` (string)
  - `year_start` (int)
  - `year_end` (int)
- Optional `weights` object
  - `organ`, `biomarker`, `drug`, `stance`, `text`, `time`, `structure`, `interaction` (float)
  - `cold_start_confidence`, `cold_start_recency`, `cold_start_entity` (float)
- Optional `flags` object
  - `exclude_unclear`, `require_all_filters`, `respect_hide_feedback` (bool)
  - `force_cold_start`, `cold_start_diversify_by_stance` (bool)
- Optional `feedback_events` list
  - each event requires `paper_id`, `feedback_type`

## Response JSON

The response contains:

- `user_id`
- `top_k`
- `count`
- `mode` (`cold_start` or `personalized`)
- `recommendations` list with rank, paper metadata, stance, score components, matched fields.

## Cold-Start Call

For a first-time user (no profile/query yet), use:

```bash
.venv/bin/python "script/recommendation/content_based_recommender.py" \
  --api-request-json "script/recommendation/recommender_api_request_cold_start_example.json" \
  --api-response-json "output_csvs/recommender_api_response_cold_start.json" \
  --feedback-state-json "output_csvs/recommender_feedback_state.json" \
  --disable-csv-output
```

This triggers cold-start scoring and stance-diversified ranking.

## Debug Mode for Fast Iteration

To run with limited candidate papers during development:

```bash
.venv/bin/python "script/recommendation/content_based_recommender.py" \
  --api-request-json "script/recommendation/recommender_api_request_example.json" \
  --api-response-json "output_csvs/recommender_api_response.json" \
  --feedback-state-json "output_csvs/recommender_feedback_state.json" \
  --disable-csv-output \
  --debug-max-papers 500
```
