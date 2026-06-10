"""
Tests for the engram raw retain endpoint and the document units sequence endpoint.

Covers the engram-raw/v1 contract guarantees:
- idempotent re-sends: units whose (document_id, line_index, sha256) already
  exist are skipped and reported as `duplicates`, never double-inserted
- sequence reconstruction: units of a session document come back ordered by
  the NUMERIC line_index (metadata values are strings, so a lexicographic
  sort would break past line 9), with around/radius and from/to windowing
"""

from datetime import datetime

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app


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
        resp = await api_client.get(
            f"/v1/engram/banks/{bank_id}/documents/{doc}/units", params=params
        )
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
    resp = await api_client.get(
        f"/v1/engram/banks/{bank_id}/documents/claude-code:none:none/units"
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 0
