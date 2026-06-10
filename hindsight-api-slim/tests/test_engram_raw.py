"""
Tests for the engram raw retain endpoint and the document units sequence endpoint.

Covers the engram-raw/v1 contract guarantees:
- idempotent re-sends: units whose (document_id, line_index, sha256) already
  exist are skipped and reported as `duplicates`, never double-inserted
- sequence reconstruction: units of a session document come back ordered by
  the NUMERIC line_index (metadata values are strings, so a lexicographic
  sort would break past line 9), with around/radius and from/to windowing
"""

import asyncio
from datetime import datetime

import asyncpg
import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.api.engram_raw import _row_to_unit, merge_hit_windows


@pytest_asyncio.fixture
async def api_client(memory):
    """Async test client for the FastAPI app (mock LLM, real embeddings)."""
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def bank_id():
    """Unique bank per test run — the session-scoped DB persists across tests."""
    return f"engram_raw_test_{datetime.now().timestamp()}"


def _make_items():
    # line indexes straddle the 9 -> 10 boundary on purpose: a lexicographic
    # ordering would yield "10" < "9" and the sequence assertions would fail.
    return [
        {
            "text": "user asks to fix the build",
            "timestamp": "2026-06-10T10:00:00Z",
            "tags": ["raw", "role:user"],
            "metadata": {"line_index": "2", "turn": "2", "role": "user"},
        },
        {
            "text": "assistant inspects the build files",
            "timestamp": "2026-06-10T10:00:05Z",
            "tags": ["raw", "role:assistant"],
            "metadata": {"line_index": "9", "turn": "2", "role": "assistant"},
        },
        {
            "text": "⏺ Bash(Build)\n$ make\n→ FATAL: undefined symbol _foo",
            "timestamp": "2026-06-10T10:00:09Z",
            "tags": ["raw", "tool-action", "tool:bash", "error"],
            "metadata": {"line_index": "10", "turn": "2", "role": "tool", "tool": "Bash"},
        },
        {
            "text": "assistant fixes the symbol and rebuilds",
            "timestamp": "2026-06-10T10:00:20Z",
            "tags": ["raw", "role:assistant"],
            "metadata": {"line_index": "11", "turn": "2", "role": "assistant"},
        },
    ]


async def _post_raw(client, bank_id, document_id, items):
    resp = await client.post(
        f"/v1/engram/banks/{bank_id}/raw",
        json={
            "items": items,
            "document_id": document_id,
            "context": "claude_code",
            "fact_type": "experience",
            "document_tags": ["raw", "agent-session"],
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def test_engram_raw_resend_is_idempotent(api_client, bank_id):
    """A re-sent slice (cursor retry after timeout/crash) must not duplicate rows."""
    doc = "claude-code:testhost:resend-session"
    items = _make_items()

    first = await _post_raw(api_client, bank_id, doc, items)
    assert first["count"] == len(items)
    assert first["duplicates"] == 0

    # full re-send: everything already stored -> nothing inserted, all reported
    again = await _post_raw(api_client, bank_id, doc, items)
    assert again["count"] == 0
    assert again["duplicates"] == len(items)
    assert again["unit_ids"] == []

    # same TEXT at a NEW line is legitimate data (e.g. the user repeats "ok"),
    # only (line_index, sha256) repeats are deduped
    repeat = dict(_make_items()[0])
    repeat["metadata"] = {**repeat["metadata"], "line_index": "40", "turn": "40"}
    third = await _post_raw(api_client, bank_id, doc, [repeat])
    assert third["count"] == 1
    assert third["duplicates"] == 0


def test_merge_hit_windows_merges_overlapping_and_touching():
    """Close hits in one document collapse into one window (no duplicated units)."""
    windows = merge_hit_windows([("doc-a", 10), ("doc-a", 14)], radius=5)
    assert windows == [{"document_id": "doc-a", "from_line": 5, "to_line": 19, "hit_lines": [10, 14]}]
    # touching windows (gap of exactly 1 line) merge too
    windows = merge_hit_windows([("doc-a", 0), ("doc-a", 7)], radius=3)
    assert windows == [{"document_id": "doc-a", "from_line": 0, "to_line": 10, "hit_lines": [0, 7]}]
    # a late hit bridging two separate windows folds them into one
    windows = merge_hit_windows([("doc-a", 0), ("doc-a", 20), ("doc-a", 10)], radius=6)
    assert windows == [{"document_id": "doc-a", "from_line": 0, "to_line": 26, "hit_lines": [0, 10, 20]}]
    # one line genuinely missing between windows (gap of 1) -> NO merge
    windows = merge_hit_windows([("doc-a", 0), ("doc-a", 10)], radius=4)
    assert [(w["from_line"], w["to_line"]) for w in windows] == [(0, 4), (6, 14)]


def test_merge_hit_windows_keeps_distinct_windows_apart():
    """Distinct documents never merge; distant hits stay separate windows in
    hit-rank order; radius 0 degenerates to the hit line; lower bound clamps at 0."""
    windows = merge_hit_windows([("doc-b", 50), ("doc-a", 2), ("doc-b", 5)], radius=3)
    assert windows == [
        {"document_id": "doc-b", "from_line": 47, "to_line": 53, "hit_lines": [50]},
        {"document_id": "doc-a", "from_line": 0, "to_line": 5, "hit_lines": [2]},
        {"document_id": "doc-b", "from_line": 2, "to_line": 8, "hit_lines": [5]},
    ]
    assert merge_hit_windows([("doc-a", 7)], radius=0) == [
        {"document_id": "doc-a", "from_line": 7, "to_line": 7, "hit_lines": [7]}
    ]
    assert merge_hit_windows([], radius=5) == []


async def test_engram_recall_expands_neighborhood(api_client, bank_id):
    """A recall hit on a raw unit comes back with its conversational window
    (small-to-big retrieval), units in numeric line order, hit text verbatim."""
    doc = "claude-code:testhost:recall-session"
    await _post_raw(api_client, bank_id, doc, _make_items())

    resp = await api_client.post(
        f"/v1/engram/banks/{bank_id}/recall",
        json={"query": "undefined symbol build failure", "radius": 1},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["results"], "expected at least one recall hit"
    hit_lines = {
        int(r["metadata"]["line_index"])
        for r in body["results"]
        if r["document_id"] == doc and "line_index" in (r["metadata"] or {})
    }
    assert hit_lines, "hits must carry document_id + metadata.line_index"

    assert body["neighborhoods"], "hits with line_index must produce neighborhoods"
    for n in body["neighborhoods"]:
        lines = [u["line_index"] for u in n["units"]]
        assert lines == sorted(lines)  # transcript order
        assert set(n["hit_lines"]) <= set(lines)  # window contains its hits
    # the window of a hit spans its radius-1 neighbors that exist in the doc
    covered = {line for n in body["neighborhoods"] for line in range(n["from_line"], n["to_line"] + 1)}
    assert any(line - 1 in covered and line + 1 in covered for line in hit_lines)


def test_row_to_unit_truncates_neighbors_not_hits():
    """Neighbor units beyond neighbor_max_chars are head-truncated and flagged;
    a unit whose line is a recall hit always keeps its full verbatim text.
    (Pure-function test: in a tiny bank every unit IS a hit, so the integration
    path cannot exercise truncation deterministically.)"""
    row = {
        "id": "unit-1",
        "text": "x" * 700,
        "tags": ["raw"],
        "metadata": {"line_index": "4", "role": "assistant"},
        "event_date": None,
    }
    neighbor = _row_to_unit(row, hit_lines={5}, neighbor_max_chars=100)
    assert neighbor["truncated"] is True
    assert neighbor["text"] == "x" * 100 + "…"

    hit = _row_to_unit(row, hit_lines={4}, neighbor_max_chars=100)
    assert "truncated" not in hit
    assert hit["text"] == "x" * 700

    untouched = _row_to_unit(row)  # sequence endpoint path: no truncation at all
    assert "truncated" not in untouched
    assert untouched["text"] == "x" * 700


async def test_engram_raw_concurrent_resend_no_duplicates(api_client, bank_id):
    """Two concurrent sends of the same slice (live flush + sweep overlapping)
    must store each unit exactly once. The old WHERE NOT EXISTS probe raced
    under READ COMMITTED; the ON CONFLICT arbiter (partial unique index,
    migration f3b8d1a6c2e9) is race-proof by construction."""
    doc = "claude-code:testhost:concurrent-session"
    items = _make_items()

    first, second = await asyncio.gather(
        _post_raw(api_client, bank_id, doc, items),
        _post_raw(api_client, bank_id, doc, items),
    )
    assert first["count"] + second["count"] == len(items)
    assert first["duplicates"] + second["duplicates"] == len(items)

    resp = await api_client.get(f"/v1/engram/banks/{bank_id}/documents/{doc}/units")
    assert resp.status_code == 200
    assert resp.json()["count"] == len(items)


async def test_engram_unique_index_blocks_direct_duplicate(api_client, bank_id, memory):
    """The DB itself rejects an exact (bank, doc, line_index, sha256) duplicate,
    independent of the endpoint's SQL — defense in depth for any future write path."""
    doc = "claude-code:testhost:backstop-session"
    await _post_raw(api_client, bank_id, doc, [_make_items()[0]])

    async with memory._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT text, metadata FROM memory_units WHERE bank_id = $1 AND document_id = $2",
            bank_id,
            doc,
        )
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(
                "INSERT INTO memory_units"
                " (bank_id, document_id, text, fact_type, metadata)"
                " VALUES ($1, $2, $3, 'experience', $4)",
                bank_id,
                doc,
                row["text"],
                row["metadata"],
            )


async def test_engram_raw_items_without_line_index_not_deduped(api_client, bank_id):
    """Free-form items (no line_index) are outside the transcript contract: stored as-is."""
    doc = "claude-code:testhost:freeform-session"
    item = {"text": "free-form note", "tags": ["raw"], "metadata": {"role": "user"}}

    first = await _post_raw(api_client, bank_id, doc, [item])
    second = await _post_raw(api_client, bank_id, doc, [item])
    assert first["count"] == 1
    assert second["count"] == 1  # no line_index -> dedupe guard does not apply
    assert second["duplicates"] == 0


async def test_engram_document_units_numeric_order_and_windows(api_client, bank_id):
    """Units come back in numeric line order; around/from/to window the sequence."""
    doc = "claude-code:testhost:sequence-session"
    await _post_raw(api_client, bank_id, doc, _make_items())

    async def get_units(**params):
        resp = await api_client.get(f"/v1/engram/banks/{bank_id}/documents/{doc}/units", params=params)
        assert resp.status_code == 200, resp.text
        return resp.json()

    # whole document: numeric order 2 < 9 < 10 < 11 (lexicographic would be 10,11,2,9)
    full = await get_units()
    assert [u["line_index"] for u in full["units"]] == [2, 9, 10, 11]
    assert full["count"] == 4

    # metadata is surfaced flat for sequence consumers
    tool_unit = next(u for u in full["units"] if u["line_index"] == 10)
    assert tool_unit["role"] == "tool"
    assert tool_unit["tool"] == "Bash"
    assert tool_unit["turn"] == "2"
    assert "FATAL" in tool_unit["text"]

    # small-to-big expansion around a recall hit
    around = await get_units(around=10, radius=1)
    assert [u["line_index"] for u in around["units"]] == [9, 10, 11]

    # explicit range
    window = await get_units(from_line=10, to_line=99)
    assert [u["line_index"] for u in window["units"]] == [10, 11]

    # unknown document -> empty, not an error
    resp = await api_client.get(f"/v1/engram/banks/{bank_id}/documents/claude-code:none:none/units")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0
