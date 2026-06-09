#!/usr/bin/env python3
"""engram ingest_raw — embedding_only capture layer (raw + embedding, zero LLM).

Reads a Claude Code session JSONL, extracts user/assistant message text
verbatim, embeds locally with BAAI/bge-m3 (1024d, same model and call path
as the running Hindsight API), and writes documents + memory_units directly
to Hindsight's Postgres. No LLM is involved at any step: the stored text is
the immutable ground truth, sha256-sealed in metadata for audit and
non-repudiation. Derived layers (facts, observations) can be regenerated
offline from these rows at any time.

Schema contract: engram-raw/v1 (see README.md).

Usage:
  ingest_raw.py ingest --session <path.jsonl> [--bank engram-test] [--limit 20]
                       [--db <dsn>] [--api <url>] [--dry-run] [--force]
  ingest_raw.py verify [--bank engram-test] [--db <dsn>] [--sample 3]
"""

import argparse
import hashlib
import json
import socket
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tags import build_tags, dedupe_preserve_order

DEFAULT_DB = "postgresql://hindsight:hindsight@127.0.0.1:5432/hindsight"
DEFAULT_API = "http://127.0.0.1:8888"
DEFAULT_BANK = "engram-test"
MODEL_NAME = "BAAI/bge-m3"
MODEL_MAX_TOKENS = 8192
SOURCE = "claude-code"
CONTEXT = "claude_code"
SCHEMA_VERSION = "engram-raw/v1"
# same noise filter as ~/.claude/hooks/hindsight-retain.py extract_new_text()
SKIP_PREFIXES = ("<command-name>", "<local-command", "<system-reminder", "Caveat:")
TSVECTOR_MAX_CHARS = 200_000  # stay far below Postgres' 1MB tsvector limit

INSERT_DOCUMENT = """
INSERT INTO documents (id, bank_id, original_text, content_hash, retain_params, tags)
VALUES (%s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT (id, bank_id) DO NOTHING
"""

# search_vector is NOT a generated column in current Hindsight (the API
# populates it at insert time) — without it the BM25 leg of recall would
# never see these rows.
INSERT_UNITS = """
INSERT INTO memory_units
  (bank_id, document_id, text, embedding, context, fact_type, tags, metadata,
   event_date, mentioned_at, search_vector)
VALUES %s
RETURNING id
"""
UNIT_TEMPLATE = (
    "(%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s,"
    " to_tsvector('english', left(%s, 200000)))"
)


@dataclass
class Msg:
    role: str
    text: str
    sha256: str
    timestamp: datetime | None
    cwd: str
    session_id: str | None
    line_index: int
    message_index: int


def parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def extract_text(content):
    """Verbatim message text: plain string, or joined text blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p).strip()
    return ""


def parse_session(path, limit):
    msgs = []
    last_hash = None
    with open(path, encoding="utf-8") as fh:
        for line_index, line in enumerate(fh, start=1):
            if len(msgs) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") not in ("user", "assistant"):
                continue
            if event.get("isMeta") or event.get("isSidechain"):
                continue
            text = extract_text(event.get("message", {}).get("content"))
            if not text or text.startswith(SKIP_PREFIXES):
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest == last_hash:
                continue
            last_hash = digest
            msgs.append(
                Msg(
                    role=event["message"]["role"],
                    text=text,
                    sha256=digest,
                    timestamp=parse_ts(event.get("timestamp")),
                    cwd=event.get("cwd", ""),
                    session_id=event.get("sessionId"),
                    line_index=line_index,
                    message_index=len(msgs),
                )
            )
    return msgs


def load_model():
    from sentence_transformers import SentenceTransformer

    # bge-m3's ST pipeline includes a Normalize module, so plain encode()
    # yields L2-normalized vectors — identical to LocalSTEmbeddings.encode.
    return SentenceTransformer(MODEL_NAME)


def embed_messages(model, msgs):
    vectors = model.encode(
        [m.text for m in msgs], convert_to_numpy=True, show_progress_bar=False
    )
    truncated = [
        len(model.tokenizer(m.text)["input_ids"]) > MODEL_MAX_TOKENS for m in msgs
    ]
    return vectors, truncated


def ensure_bank(api, bank):
    """Create the bank via the API (also pre-builds per-bank vector indexes)."""
    req = urllib.request.Request(
        f"{api}/v1/default/banks/{bank}",
        data=json.dumps({"name": bank}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status
    except Exception as exc:  # non-fatal: writes don't need the banks row
        print(f"warning: bank ensure failed ({exc}); relying on lazy creation",
              file=sys.stderr)
        return None


def cmd_ingest(args):
    session_path = Path(args.session).resolve()
    msgs = parse_session(session_path, args.limit)
    if not msgs:
        sys.exit("no ingestible messages found")

    session_id = msgs[0].session_id or session_path.stem
    host = socket.gethostname().split(".")[0].lower()
    document_id = f"{SOURCE}:{host}:{session_id}"
    project = Path(msgs[0].cwd).name or "unknown"
    per_msg_tags = [
        build_tags(source=SOURCE, host=host, project=project, role=m.role, text=m.text)
        for m in msgs
    ]

    print(f"session  {session_id}")
    print(f"document {document_id}  bank {args.bank}  messages {len(msgs)}")
    for m, tags in zip(msgs, per_msg_tags):
        print(f"  #{m.message_index:>2} L{m.line_index:>5} {m.role:<9} "
              f"{len(m.text):>6} chars  {','.join(tags)}")
    if args.dry_run:
        print("dry-run: nothing written")
        return

    print(f"loading {MODEL_NAME} …")
    model = load_model()
    vectors, truncated = embed_messages(model, msgs)
    for tags, trunc in zip(per_msg_tags, truncated):
        if trunc:
            tags[:] = dedupe_preserve_order([*tags, "truncated-embedding"])

    original_text = "\n\n".join(f"[{m.role}] {m.text}" for m in msgs)
    doc_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
    doc_tags = dedupe_preserve_order(
        [t for tags in per_msg_tags for t in tags if not t.startswith("role:")]
    )
    ingested_at = datetime.now(timezone.utc).isoformat()
    retain_params = {
        "engram": {
            "mode": "embedding_only",
            "schema": SCHEMA_VERSION,
            "source_path": str(session_path),
            "limit": args.limit,
            "ingested_at": ingested_at,
        }
    }

    ensure_bank(args.api, args.bank)

    import psycopg2
    from psycopg2.extras import execute_values
    from pgvector.psycopg2 import register_vector

    conn = psycopg2.connect(args.db)
    register_vector(conn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                INSERT_DOCUMENT,
                (document_id, args.bank, original_text, doc_hash,
                 json.dumps(retain_params), doc_tags),
            )
            if cur.rowcount == 0:
                if not args.force:
                    sys.exit(f"document {document_id} already exists in bank "
                             f"{args.bank}; use --force to replace")
                cur.execute(
                    "DELETE FROM documents WHERE id = %s AND bank_id = %s",
                    (document_id, args.bank),
                )
                cur.execute(
                    INSERT_DOCUMENT,
                    (document_id, args.bank, original_text, doc_hash,
                     json.dumps(retain_params), doc_tags),
                )
            rows = []
            for m, tags, vector, trunc in zip(msgs, per_msg_tags, vectors, truncated):
                # all values MUST be strings: recall validates metadata as
                # Dict[str, str] (MemoryFact) — non-string values make the
                # bank unsearchable (empty results or 500 ValidationError)
                metadata = {
                    "schema": SCHEMA_VERSION,
                    "sha256": m.sha256,
                    "source": SOURCE,
                    "host": host,
                    "session_id": session_id,
                    "source_path": str(session_path),
                    "message_index": str(m.message_index),
                    "line_index": str(m.line_index),
                    "role": m.role,
                    "ingested_at": ingested_at,
                    "embedding_truncated": "true" if trunc else "false",
                }
                rows.append(
                    (args.bank, document_id, m.text, vector, CONTEXT, "experience",
                     tags, json.dumps(metadata), m.timestamp, m.timestamp,
                     (m.text + " " + CONTEXT)[:TSVECTOR_MAX_CHARS])
                )
            ids = execute_values(cur, INSERT_UNITS, rows,
                                 template=UNIT_TEMPLATE, fetch=True)
        print(f"ingested {len(ids)} memory_units into bank {args.bank} "
              f"(document {document_id})")
    finally:
        conn.close()


def cmd_verify(args):
    import numpy as np
    import psycopg2
    from pgvector.psycopg2 import register_vector

    conn = psycopg2.connect(args.db)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT text, embedding, metadata, tags, document_id, context,
               search_vector IS NOT NULL
        FROM memory_units
        WHERE bank_id = %s
        ORDER BY (metadata->>'message_index')::int
        """,
        (args.bank,),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        sys.exit(f"no rows in bank {args.bank}")

    hash_fail = [
        i for i, (text, _, meta, *_rest) in enumerate(rows)
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != meta.get("sha256")
    ]
    missing_sv = sum(1 for r in rows if not r[6])
    missing_emb = sum(1 for r in rows if r[1] is None)
    norms = [float(np.linalg.norm(r[1])) for r in rows if r[1] is not None]

    print(f"bank {args.bank}: {len(rows)} units")
    print(f"  sha256 integrity : {len(rows) - len(hash_fail)}/{len(rows)} match")
    print(f"  embeddings       : {len(rows) - missing_emb}/{len(rows)} present, "
          f"avg L2 norm {sum(norms)/len(norms):.4f}")
    print(f"  search_vector    : {len(rows) - missing_sv}/{len(rows)} populated")
    print(f"  document_id      : {rows[0][4]}")
    print(f"  context          : {rows[0][5]}")

    print(f"re-encoding {min(args.sample, len(rows))} sample(s) with {MODEL_NAME} …")
    model = load_model()
    worst = 1.0
    for text, stored, *_rest in rows[: args.sample]:
        fresh = model.encode([text], convert_to_numpy=True)[0]
        cos = float(np.dot(fresh, stored)
                    / (np.linalg.norm(fresh) * np.linalg.norm(stored)))
        worst = min(worst, cos)
        print(f"  cosine(fresh, stored) = {cos:.6f}")

    ok = not hash_fail and not missing_sv and not missing_emb and worst > 0.999
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest a session JSONL")
    p_ingest.add_argument("--session", required=True)
    p_ingest.add_argument("--bank", default=DEFAULT_BANK)
    p_ingest.add_argument("--limit", type=int, default=20)
    p_ingest.add_argument("--db", default=DEFAULT_DB)
    p_ingest.add_argument("--api", default=DEFAULT_API)
    p_ingest.add_argument("--dry-run", action="store_true")
    p_ingest.add_argument("--force", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_verify = sub.add_parser("verify", help="verify integrity of an ingested bank")
    p_verify.add_argument("--bank", default=DEFAULT_BANK)
    p_verify.add_argument("--db", default=DEFAULT_DB)
    p_verify.add_argument("--sample", type=int, default=3)
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
