"""engram fork: raw embedding_only retain endpoint — zero LLM.

Stores agent-session items verbatim (immutable text + server-side sha256)
with an embedding computed by the API's resident embedding model, plus an
explicit search_vector for the BM25 leg of recall. The write path never
touches an LLM: this is the ground-truth capture layer for audit-grade
memory (non-repudiation); derived layers (facts, observations, mental
models) are regenerable offline from these rows.

Fork-owned file: everything engram lives here except a 3-line registration
hook in http.py (minimal upstream delta — see fork strategy notes).

Contract: engram-raw/v1 (see engram/README.md at the repo root).
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone

from fastapi import HTTPException
from pydantic import BaseModel, Field

from ..config import get_config
from ..engine.memory_engine import fq_table
from ..models import RequestContext

ENGRAM_SCHEMA = "engram-raw/v1"
TSVECTOR_MAX_CHARS = 200_000  # stay far below Postgres' 1MB tsvector limit
# Attention is O(n^2) in sequence length and sentence-transformers pads the
# whole batch to the longest text: a batch of near-8192-token items asks MPS
# for ~40GiB attention buffers and crashes the request. Cap the EMBEDDING
# input only (stored text stays full/verbatim, sha256 unchanged) and encode
# in small sub-batches.
EMBED_CHAR_CAP = 6_000
EMBED_SUB_BATCH = 8

# Deterministic topic tagging (V1 taxonomy), applied server-side so every
# client (stormwind hook, M4 over Tailscale, codex, hermes-agent) gets the
# same taxonomy without shipping it. Case-insensitive substring matching —
# zero LLM. Keep in sync with engram/tags.py (standalone batch ingest).
TOPIC_KEYWORDS = {
    "hindsight": ["hindsight"],
    "engram": ["engram"],
    "memory-system": ["retain", "recall", "memory bank", "memory system", "consolidat"],
    "embedding": ["embedding", "bge-m3", "sentence-transformers", "pgvector", "vector("],
    "postgres": ["postgres", "postgresql", "psql"],
    "llm": ["llm", "llama.cpp", "ollama", "qwen", "claude", "gpt"],
    "rust": ["rust", "cargo", "pyo3", "maturin"],
    "python": ["python", "venv", "pip install"],
    "git": ["git ", "worktree", "rebase", "branch"],
    "macos": ["macos", "launchd", "darwin", "mac studio", "m1 max"],
    "obsidian": ["obsidian", "vault"],
    "networking": ["tailscale", "ssh "],
    "audit": ["sha256", "sha-256", "non-repudiation", "immutable"],
}


def detect_topic_tags(text: str) -> list[str]:
    haystack = text.lower()
    return [tag for tag, keywords in TOPIC_KEYWORDS.items() if any(keyword in haystack for keyword in keywords)]


def dedupe_preserve_order(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


class EngramRawItem(BaseModel):
    text: str = Field(description="Verbatim message text — stored unmodified")
    timestamp: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    # string values only: recall validates metadata as Dict[str, str]
    metadata: dict[str, str] = Field(default_factory=dict)


class EngramRawRequest(BaseModel):
    items: list[EngramRawItem]
    document_id: str
    context: str = "claude_code"
    fact_type: str = "experience"
    document_tags: list[str] = Field(default_factory=list)


INSERT_DOCUMENT = """
INSERT INTO {documents} (id, bank_id, original_text, content_hash, retain_params, tags)
VALUES ($1, $2, '', $3, $4::jsonb, $5)
ON CONFLICT (id, bank_id) DO NOTHING
"""

# search_vector is populated explicitly: it is NOT a generated column, and a
# row without it is invisible to the BM25 leg of recall.
# Idempotent re-sends: a unit whose (document_id, line_index, sha256) already
# exists is skipped — the client cursor protocol (commit-on-success) re-sends a
# whole slice after a timeout/crash, and without this guard every retry would
# duplicate the rows it had in fact stored. Only applies to items that carry
# line_index (the agent-transcript contract); free-form items are not deduped.
INSERT_UNITS = """
WITH input AS (
    SELECT * FROM unnest(
        $3::text[], $4::vector[], $5::timestamptz[], $6::jsonb[], $7::jsonb[]
    ) AS t(text, embedding, ts, tags_json, metadata)
)
INSERT INTO {memory_units}
    (bank_id, document_id, text, embedding, context, fact_type, tags, metadata,
     event_date, mentioned_at, search_vector)
SELECT
    $1, $2, text, embedding, $8, $9,
    COALESCE((SELECT array_agg(e) FROM jsonb_array_elements_text(tags_json) AS e),
             '{{}}'::varchar[]),
    metadata, ts, ts,
    to_tsvector('{language}'::regconfig,
                left(COALESCE(text, '') || ' ' || $8, {max_chars}))
FROM input
WHERE NOT (input.metadata ? 'line_index')
   OR NOT EXISTS (
        SELECT 1 FROM {memory_units} existing
        WHERE existing.bank_id = $1
          AND existing.document_id = $2
          AND existing.metadata ? 'sha256'
          AND existing.metadata->>'sha256' = input.metadata->>'sha256'
          AND existing.metadata->>'line_index' = input.metadata->>'line_index'
   )
RETURNING id
"""

# Sequence reconstruction: raw units of one session document, ordered by their
# transcript line. The ::int cast is REQUIRED (metadata values are strings, and
# "10" < "9" lexicographically) and is what the partial expression index
# idx_memory_units_engram_doc_line is built on.
SELECT_DOCUMENT_UNITS = """
SELECT id, text, tags, metadata, event_date
FROM {memory_units}
WHERE bank_id = $1 AND document_id = $2
  AND metadata ? 'line_index'
  AND (metadata->>'line_index')::int BETWEEN $3 AND $4
ORDER BY (metadata->>'line_index')::int
LIMIT $5
"""


def register_engram_raw_routes(app) -> None:
    @app.post(
        "/v1/engram/banks/{bank_id}/raw",
        summary="engram raw retain (embedding_only, zero LLM)",
        description=(
            "Stores items verbatim with server-side sha256, resident-model "
            "embedding and explicit search_vector. Synchronous — no async "
            "operation, no LLM, no polling."
        ),
        tags=["engram"],
    )
    async def engram_raw_retain(bank_id: str, request: EngramRawRequest):
        memory = app.state.memory
        items = [item for item in request.items if item.text.strip()]
        if not items:
            return {
                "bank_id": bank_id,
                "document_id": request.document_id,
                "unit_ids": [],
                "count": 0,
                "timings_ms": {},
            }

        config = get_config()
        if config.text_search_extension != "native":
            raise HTTPException(
                status_code=500,
                detail="engram raw retain supports text_search_extension=native only",
            )

        texts = [item.text for item in items]
        embed_texts = [t[:EMBED_CHAR_CAP] for t in texts]
        truncated = [len(t) > EMBED_CHAR_CAP for t in texts]
        t0 = time.perf_counter()
        # resident model, off the event loop (MPS encode can take 100s of ms).
        # Length-sort before sub-batching so short texts batch together
        # (padding waste is per-batch: one long text pads its whole batch).
        order = sorted(range(len(embed_texts)), key=lambda i: len(embed_texts[i]))
        sorted_vectors: list = []
        for off in range(0, len(order), EMBED_SUB_BATCH):
            chunk = [embed_texts[i] for i in order[off : off + EMBED_SUB_BATCH]]
            sorted_vectors.extend(await asyncio.to_thread(memory.embeddings.encode_documents, chunk))
        vectors: list = [None] * len(embed_texts)
        for rank, i in enumerate(order):
            vectors[i] = sorted_vectors[rank]
        t_embed = time.perf_counter() - t0

        ingested_at = datetime.now(timezone.utc).isoformat()
        embeddings_str: list[str] = []
        timestamps: list[datetime | None] = []
        tags_jsons: list[str] = []
        metadata_jsons: list[str] = []
        for item, vector, trunc in zip(items, vectors, truncated):
            embeddings_str.append(str([float(x) for x in vector]))
            timestamps.append(item.timestamp)
            unit_tags = [*item.tags, *detect_topic_tags(item.text)]
            if trunc:
                unit_tags.append("truncated-embedding")
            tags_jsons.append(json.dumps(dedupe_preserve_order(unit_tags)))
            metadata = dict(item.metadata)
            # sha256 sealed server-side over the exact stored text
            metadata["sha256"] = hashlib.sha256(item.text.encode("utf-8")).hexdigest()
            metadata.setdefault("schema", ENGRAM_SCHEMA)
            metadata["ingested_at"] = ingested_at
            if trunc:
                metadata["embedding_truncated"] = "true"
            metadata_jsons.append(json.dumps(metadata))

        insert_doc = INSERT_DOCUMENT.format(documents=fq_table("documents"))
        insert_units = INSERT_UNITS.format(
            memory_units=fq_table("memory_units"),
            language=config.text_search_extension_native_language,
            max_chars=TSVECTOR_MAX_CHARS,
        )

        t1 = time.perf_counter()
        async with memory._pool.acquire() as conn:
            async with conn.transaction():
                await memory._ensure_bank_exists(bank_id, RequestContext(internal=True), conn=conn)
                await conn.execute(
                    insert_doc,
                    request.document_id,
                    bank_id,
                    ENGRAM_SCHEMA,
                    json.dumps({"engram": {"mode": "embedding_only", "schema": ENGRAM_SCHEMA}}),
                    request.document_tags,
                )
                rows = await conn.fetch(
                    insert_units,
                    bank_id,
                    request.document_id,
                    texts,
                    embeddings_str,
                    timestamps,
                    tags_jsons,
                    metadata_jsons,
                    request.context,
                    request.fact_type,
                )
        t_insert = time.perf_counter() - t1

        return {
            "bank_id": bank_id,
            "document_id": request.document_id,
            "unit_ids": [str(row["id"]) for row in rows],
            "count": len(rows),
            # re-sent units skipped by the (document_id, line_index, sha256)
            # guard — clients must treat count + duplicates > 0 as success.
            "duplicates": len(items) - len(rows),
            "timings_ms": {
                "embed": round(t_embed * 1000, 1),
                "insert": round(t_insert * 1000, 1),
            },
        }

    @app.get(
        "/v1/engram/banks/{bank_id}/documents/{document_id}/units",
        summary="engram raw units of a document, in transcript order",
        description=(
            "Sequence reconstruction for raw agent-session units: returns the "
            "units of one session document ordered by their transcript "
            "line_index. Use around+radius to expand a recall hit to its "
            "conversational neighborhood (small-to-big retrieval), or "
            "from_line/to_line for an explicit range."
        ),
        tags=["engram"],
    )
    async def engram_document_units(
        bank_id: str,
        document_id: str,
        around: int | None = None,
        radius: int = 10,
        from_line: int | None = None,
        to_line: int | None = None,
        limit: int = 200,
    ):
        memory = app.state.memory
        limit = max(1, min(limit, 1000))
        if around is not None:
            lo, hi = max(0, around - radius), around + radius
        else:
            lo = from_line if from_line is not None else 0
            hi = to_line if to_line is not None else 2**31 - 1
        sql = SELECT_DOCUMENT_UNITS.format(memory_units=fq_table("memory_units"))
        async with memory._pool.acquire() as conn:
            rows = await conn.fetch(sql, bank_id, document_id, lo, hi, limit)
        units = []
        for row in rows:
            raw_md = row["metadata"]
            md = json.loads(raw_md) if isinstance(raw_md, str) else (raw_md or {})
            units.append(
                {
                    "id": str(row["id"]),
                    "line_index": int(md.get("line_index", -1)),
                    "role": md.get("role"),
                    "tool": md.get("tool"),
                    "turn": md.get("turn"),
                    "tags": list(row["tags"] or []),
                    "timestamp": row["event_date"].isoformat() if row["event_date"] else None,
                    "text": row["text"],
                }
            )
        return {
            "bank_id": bank_id,
            "document_id": document_id,
            "count": len(units),
            "units": units,
        }
