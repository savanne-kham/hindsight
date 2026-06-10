# engram — raw `embedding_only` capture layer

> **Two write paths, one contract (`engram-raw/v1`):**
> 1. **Live**: `POST /v1/engram/banks/{bank}/raw` (fork endpoint,
>    `hindsight_api/api/engram_raw.py`) — used by the Claude Code hook
>    (`~/.claude/hooks/hindsight-retain.py`) into bank `engram-raw`. The server
>    seals sha256, embeds with the **resident** bge-m3 (~100 ms warm, zero
>    LLM) and adds deterministic topic tags server-side, so every client
>    (stormwind, M4 over Tailscale, codex, hermes-agent) shares the taxonomy.
> 2. **Batch/offline**: `ingest_raw.py` below — direct Postgres, for bulk
>    imports and for when the API is down.

Ground-truth capture for a self-owned long-term memory system built on the
Hindsight schema. Stores agent-session messages **verbatim** (raw immutable
text + sha256) with a locally computed bge-m3 embedding — **zero LLM** in the
write path (~ms instead of seconds, and the first-instance memory is never
rewritten, which preserves non-repudiation for audit-grade use). LLM-derived
layers (facts, observations, mental models) are regenerable offline from
these rows and live elsewhere.

New files only — nothing upstream is modified, so rebases on `upstream/main`
stay conflict-free.

## Schema contract — `engram-raw/v1`

One `memory_units` row per user/assistant message:

| column | value |
|---|---|
| `text` | verbatim message text (string content, or joined `text` blocks) |
| `embedding` | `BAAI/bge-m3`, 1024d, plain `model.encode()` (L2-normalized by the model's ST pipeline — same call path as the Hindsight API) |
| `fact_type` | `experience` |
| `context` | `claude_code` |
| `document_id` | `claude-code:<host>:<session_id>` |
| `tags` | deterministic: `raw, agent-session, claude-code, <host>, project:<cwd>, role:<role>` + substring-matched topic tags (`tags.py`) |
| `metadata` | `{schema, sha256, source, host, session_id, source_path, message_index, line_index, turn, role, tool, line_span, batch_size, ingested_at, embedding_truncated}` — **string values only**: recall validates metadata as `Dict[str, str]` (`MemoryFact`); a single int/bool value makes the whole bank unsearchable. `turn` = line_index of the initiating user message (grouping key to reassemble one conversational turn); `line_span`/`batch_size` only on merged runs of sparse tool actions (tag `tool-batch`) |
| `event_date` / `mentioned_at` | message timestamp (feeds the temporal recall leg) |
| `search_vector` | `to_tsvector('english', text … )` — **must be set explicitly**: it is *not* a generated column; the API fills it at insert time, so a direct INSERT that omits it is invisible to the BM25 leg of recall |

One `documents` row per session slice: `id = document_id`,
`original_text` = concatenated `[role] text` slice, `content_hash` =
sha256(original_text), `retain_params.engram` = ingest provenance.

Filtering mirrors the live hook (`~/.claude/hooks/hindsight-retain.py`):
user/assistant only, skip `isMeta`/`isSidechain`, skip
`<command-name>`/`<local-command`/`<system-reminder`/`Caveat:` noise,
skip consecutive duplicates (sha256). Messages longer than the model's
8192-token window keep their full raw text; only the embedding is truncated
(flagged with tag `truncated-embedding` + `metadata.embedding_truncated`).

## Run

```bash
VENV=/Users/guinsoo/dev/ai-lab/hindsight-config/macstudio-m1max/venv-hindsight
cd engram   # this directory

$VENV/bin/python ingest_raw.py ingest \
  --session ~/.claude/projects/-Users-guinsoo/<session>.jsonl \
  --bank engram-test --limit 20 [--dry-run] [--force]

$VENV/bin/python ingest_raw.py verify --bank engram-test
```

Re-ingesting the same session is an idempotent no-op (document PK conflict);
`--force` atomically replaces the document and its units (FK cascade).

## Sequence + idempotence (API)

Raw units are ordered by `(document_id, (metadata->>'line_index')::int)` — a
sort key, not a linked list: the JSONL transcript is the authoritative chain,
the DB only projects it (expression index `idx_memory_units_engram_doc_line`).
Reconstruct a recall hit's neighborhood (small-to-big retrieval):

```bash
curl -s "localhost:8888/v1/engram/banks/engram-raw/documents/<doc>/units?around=216&radius=10" | jq
# or an explicit range: ?from_line=100&to_line=200   (limit ≤ 1000)
```

Idempotence is enforced BY THE DATABASE: a partial unique index on
`(bank_id, document_id, line_index, sha256)` (migration `f3b8d1a6c2e9`,
`uq_memory_units_engram_doc_line_sha`) is the `ON CONFLICT DO NOTHING`
arbiter of the raw retain INSERT. Re-sent units are skipped and reported as
`duplicates` in the response, so a slice retried after a timeout/crash never
double-inserts — race-proof even when a live flush and a reconcile sweep send
the same slice concurrently (the previous `WHERE NOT EXISTS` guard was not,
and let 27 double-inserts through before being replaced). Clients must treat
`count + duplicates > 0` as success.

## Recall with neighborhood expansion

`POST /v1/engram/banks/{bank}/recall` — hybrid recall (semantic + BM25 +
graph + temporal, reranked, zero LLM) followed by small-to-big expansion:
every hit carrying a `line_index` is expanded to its conversational window
`[line-radius, line+radius]`; overlapping/contiguous windows of one document
are merged (one neighborhood per cluster of close hits, no duplicated units).

```bash
curl -s -X POST localhost:8888/v1/engram/banks/engram-raw/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "MPS reranker wedge", "radius": 5}' | jq '.neighborhoods[0]'
```

Body: `query`, `max_tokens` (hit budget, default 4096), `radius` (default 5,
clamped 0–50, 0 = no expansion), `types`, `tags`, `neighbor_max_chars`
(default 600 — neighbor units are head-truncated and flagged
`"truncated": true`; HIT units always stay verbatim), `max_neighborhood_chars`
(default 24000 — global cap filled best-hit-first; windows beyond it return
coordinates only with `units_omitted: true`, fetchable via the units
endpoint above). Each request logs one `[ENGRAM RECALL]` line (tailed by the
`recall-log` tmux window on the llama server).

Both guarantees are covered by `tests/test_engram_raw.py`:

```bash
psql -d postgres -c "CREATE DATABASE hindsight_test OWNER hindsight" 2>/dev/null
psql -d hindsight_test -c "CREATE EXTENSION IF NOT EXISTS vector"   # needs superuser
cd hindsight-api-slim && HINDSIGHT_API_DATABASE_URL=postgresql://hindsight:hindsight@127.0.0.1:5432/hindsight_test \
  uv run --extra local-ml pytest tests/test_engram_raw.py -n 0
# -n 0: xdist workers race the migration runner on a shared external DB
```

## Validate

```bash
DSN=postgresql://hindsight:hindsight@127.0.0.1:5432/hindsight
# counts + norms (expect N/N/N and avg norm ≈ 1.0)
psql $DSN -c "SELECT count(*), count(embedding), count(search_vector),
  round(avg(vector_norm(embedding))::numeric,4)
  FROM memory_units WHERE bank_id='engram-test';"
# hybrid recall through the running API
curl -s -X POST localhost:8888/v1/default/banks/engram-test/memories/recall \
  -H 'Content-Type: application/json' \
  -d '{"query":"<distinctive phrase>","types":["experience"],"max_tokens":2048}' | jq
# zero-LLM proof
psql $DSN -c "SELECT count(*) FROM llm_requests WHERE bank_id='engram-test';"
```
