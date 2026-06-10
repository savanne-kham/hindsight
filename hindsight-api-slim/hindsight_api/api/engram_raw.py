"""engram fork: engraphy endpoint — raw embedding_only write, zero LLM.

ENGRAPHY is the write action of the engram layer (Richard Semon's own term,
1904: the process that inscribes an experience into the memory substrate —
he also coined "engram" and "ecphory", the read-side cueing, reserved here
for the recall expansion). Deliberately NOT called "retain": Hindsight's
retain is the LLM fact-extraction pipeline; engraphy is its zero-LLM,
verbatim counterpart.

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
import logging
import time
from datetime import datetime, timezone

from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

from ..config import get_config
from ..engine.memory_engine import fq_table
from ..engine.response_models import VALID_RECALL_FACT_TYPES
from ..models import RequestContext

logger = logging.getLogger(__name__)

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


class EngramRecallRequest(BaseModel):
    query: str
    max_tokens: int = Field(default=4096, description="Token budget for the recall hits themselves")
    # raw units are messages; a conversational turn spans 2-15 of them, so ±5
    # captures the median enclosing turn. Overlap-merging keeps dense hit
    # clusters cheap. Clamped to 0..50 (0 = no expansion).
    radius: int = 5
    types: list[str] | None = None
    tags: list[str] | None = None
    # context units are head-truncated; recall hits always stay verbatim.
    neighbor_max_chars: int = 600
    # global cap over all neighborhood unit texts, filled best-hit-first;
    # windows beyond it return coordinates only (units_omitted=true).
    max_neighborhood_chars: int = 24_000


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
# duplicate the rows it had in fact stored. The guard is the partial UNIQUE
# index uq_memory_units_engram_doc_line_sha (migration f3b8d1a6c2e9) used as
# ON CONFLICT arbiter: unlike the previous WHERE NOT EXISTS probe it is
# race-proof under concurrent re-sends (live flush + sweep overlapping) and
# also dedupes identical rows within one batch. Only items that carry
# line_index (the agent-transcript contract) match the index predicate;
# free-form items are not deduped. RETURNING reports inserted rows only, so
# the `duplicates` accounting in the response stays exact.
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
ON CONFLICT (bank_id, document_id, (metadata->>'line_index'), (metadata->>'sha256'))
    WHERE metadata ? 'line_index' AND metadata ? 'sha256'
    DO NOTHING
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


def merge_hit_windows(hits: list[tuple[str, int]], radius: int) -> list[dict]:
    """(document_id, line_index) pairs -> merged per-document line windows.

    Each hit expands to [line-radius, line+radius]; windows of the same
    document that overlap or are contiguous (no transcript line missing
    between them) are merged so that close hits — adjacent messages of one
    conversation — yield ONE neighborhood instead of duplicated unit lists.
    Windows keep first-hit-rank order: callers fill a char budget
    best-hit-first. Pure function (unit tested)."""
    windows: list[dict] = []
    for document_id, line in hits:
        lo, hi = max(0, line - radius), line + radius
        target = None
        keep = []
        for w in windows:
            if w["document_id"] == document_id and w["from_line"] <= hi + 1 and lo <= w["to_line"] + 1:
                if target is None:
                    target = w
                    w["from_line"] = min(w["from_line"], lo)
                    w["to_line"] = max(w["to_line"], hi)
                    w["hit_lines"].append(line)
                    keep.append(w)
                else:
                    # this hit bridges two existing windows: fold w into target
                    target["from_line"] = min(target["from_line"], w["from_line"])
                    target["to_line"] = max(target["to_line"], w["to_line"])
                    target["hit_lines"].extend(w["hit_lines"])
            else:
                keep.append(w)
        if target is None:
            keep.append({"document_id": document_id, "from_line": lo, "to_line": hi, "hit_lines": [line]})
        windows = keep
    for w in windows:
        w["hit_lines"] = sorted(set(w["hit_lines"]))
    return windows


def _row_to_unit(row, *, hit_lines: set[int] | None = None, neighbor_max_chars: int = 0) -> dict:
    """Shared unit shape of the sequence/neighborhood payloads. Neighbor units
    (not a recall hit themselves) are head-truncated to neighbor_max_chars —
    they are context, the full verbatim text stays one GET (or the JSONL
    archive) away; hit units always keep their full text."""
    raw_md = row["metadata"]
    md = json.loads(raw_md) if isinstance(raw_md, str) else (raw_md or {})
    line_index = int(md.get("line_index", -1))
    text = row["text"]
    truncated = False
    if neighbor_max_chars and len(text) > neighbor_max_chars and (hit_lines is None or line_index not in hit_lines):
        text = text[:neighbor_max_chars] + "…"
        truncated = True
    unit = {
        "id": str(row["id"]),
        "line_index": line_index,
        "role": md.get("role"),
        "tool": md.get("tool"),
        "turn": md.get("turn"),
        "tags": list(row["tags"] or []),
        "timestamp": row["event_date"].isoformat() if row["event_date"] else None,
        "text": text,
    }
    if truncated:
        unit["truncated"] = True
    return unit


def register_engram_raw_routes(app) -> None:
    @app.post(
        "/v1/engram/banks/{bank_id}/raw",
        summary="engram engraphy (raw embedding_only write, zero LLM)",
        description=(
            "Stores items verbatim with server-side sha256, resident-model "
            "embedding and explicit search_vector. Synchronous — no async "
            "operation, no LLM, no polling."
        ),
        tags=["engram"],
    )
    async def engram_engraph(bank_id: str, request: EngramRawRequest):
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
                detail="engram engraphy supports text_search_extension=native only",
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
        units = [_row_to_unit(row) for row in rows]
        return {
            "bank_id": bank_id,
            "document_id": document_id,
            "count": len(units),
            "units": units,
        }

    @app.post(
        "/v1/engram/banks/{bank_id}/recall",
        summary="engram recall with conversational-neighborhood expansion",
        description=(
            "Hybrid recall (semantic + BM25 + graph + temporal, reranked — "
            "zero LLM) over raw agent-session units, then small-to-big "
            "expansion: every hit that carries a transcript line_index is "
            "expanded to its conversational neighborhood "
            "[line-radius, line+radius]; overlapping windows of one document "
            "are merged. Neighbor units are head-truncated to "
            "neighbor_max_chars (hits stay verbatim); windows are filled "
            "best-hit-first until max_neighborhood_chars, the rest come back "
            "with units_omitted=true and can be fetched via the documents/"
            "{document_id}/units endpoint."
        ),
        tags=["engram"],
    )
    async def engram_recall(
        bank_id: str,
        request: EngramRecallRequest,
        authorization: str | None = Header(default=None),
    ):
        memory = app.state.memory
        radius = max(0, min(request.radius, 50))
        # mirror http.py's recall route: API key from the Authorization header
        # (no-op on the single-tenant deployment, correct if auth is enabled),
        # observation facts excluded unless asked for.
        api_key = None
        if authorization:
            api_key = (
                authorization[7:].strip() if authorization.lower().startswith("bearer ") else authorization.strip()
            )
        t0 = time.perf_counter()
        result = await memory.recall_async(
            bank_id,
            request.query,
            max_tokens=request.max_tokens,
            fact_type=request.types or list(VALID_RECALL_FACT_TYPES),
            tags=request.tags,
            request_context=RequestContext(api_key=api_key),
        )
        t_recall = time.perf_counter() - t0

        results, hits = [], []
        for fact in result.results:
            md = fact.metadata or {}
            results.append(
                {
                    "id": fact.id,
                    "text": fact.text,
                    "type": fact.fact_type,
                    "document_id": fact.document_id,
                    "metadata": md,
                    "tags": fact.tags,
                    "mentioned_at": fact.mentioned_at,
                }
            )
            line = md.get("line_index")
            if fact.document_id and line is not None and str(line).isdigit():
                hits.append((fact.document_id, int(line)))

        t1 = time.perf_counter()
        neighborhoods = []
        if hits and radius > 0:
            sql = SELECT_DOCUMENT_UNITS.format(memory_units=fq_table("memory_units"))
            budget = request.max_neighborhood_chars
            async with memory._pool.acquire() as conn:
                for window in merge_hit_windows(hits, radius):
                    rows = await conn.fetch(
                        sql,
                        bank_id,
                        window["document_id"],
                        window["from_line"],
                        window["to_line"],
                        window["to_line"] - window["from_line"] + 1,
                    )
                    units = [
                        _row_to_unit(
                            row,
                            hit_lines=set(window["hit_lines"]),
                            neighbor_max_chars=request.neighbor_max_chars,
                        )
                        for row in rows
                    ]
                    size = sum(len(u["text"]) for u in units)
                    if size <= budget:
                        budget -= size
                        neighborhoods.append({**window, "units": units})
                    else:
                        # budget exhausted: ship the window coordinates only —
                        # the client can still fetch it explicitly.
                        neighborhoods.append({**window, "units": [], "units_omitted": True})
        t_expand = time.perf_counter() - t1

        # one line per request, same spirit as http.py's [RECALL HTTP] —
        # this is what the recall-log tmux window tails.
        logger.info(
            "[ENGRAM RECALL] bank=%s results=%d neighborhoods=%d radius=%d recall=%.1fms expand=%.1fms",
            bank_id,
            len(results),
            len(neighborhoods),
            radius,
            t_recall * 1000,
            t_expand * 1000,
        )
        return {
            "bank_id": bank_id,
            "query": request.query,
            "results": results,
            "neighborhoods": neighborhoods,
            "timings_ms": {
                "recall": round(t_recall * 1000, 1),
                "expand": round(t_expand * 1000, 1),
            },
        }
