"""Enforce engram raw idempotence in the database: unique (doc, line, sha) index

Revision ID: f3b8d1a6c2e9
Revises: e7a1c9d2b4f6
Create Date: 2026-06-10

The raw retain endpoint deduplicated re-sent slices with a WHERE NOT EXISTS
guard. Under READ COMMITTED two concurrent sends of the same slice (live hook
flush + reconcile sweep overlapping) can both pass the existence check and
double-insert — observed live: 27 exact (document_id, line_index, sha256)
double-inserts in bank engram-raw. Application-level checks cannot close that
race; a unique index can.

Three steps, in order (each idempotent, no-ops on a clean database):

  1. Purge existing exact duplicates, keeping the earliest row per
     (bank_id, document_id, line_index, sha256) group. Raw-layer immutability
     is preserved: only surplus identical copies are removed, no text or
     metadata is touched.
  2. Create the partial UNIQUE expression index. It becomes the ON CONFLICT
     arbiter of the raw retain INSERT (see api/engram_raw.py).
  3. Drop idx_memory_units_engram_doc_sha (e7a1c9d2b4f6): its only purpose
     was serving the NOT EXISTS probe, now superseded by the unique index.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f3b8d1a6c2e9"
down_revision: str | Sequence[str] | None = "e7a1c9d2b4f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"""
        DELETE FROM {schema}memory_units m
        USING (
            SELECT id, row_number() OVER (
                PARTITION BY bank_id, document_id,
                             metadata->>'line_index', metadata->>'sha256'
                ORDER BY created_at, id
            ) AS rn
            FROM {schema}memory_units
            WHERE metadata ? 'line_index' AND metadata ? 'sha256'
        ) dup
        WHERE m.id = dup.id AND dup.rn > 1
    """)
    op.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_units_engram_doc_line_sha
        ON {schema}memory_units
           (bank_id, document_id, (metadata->>'line_index'), (metadata->>'sha256'))
        WHERE metadata ? 'line_index' AND metadata ? 'sha256'
    """)
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_engram_doc_sha")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}uq_memory_units_engram_doc_line_sha")
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_engram_doc_sha
        ON {schema}memory_units (document_id, (metadata->>'sha256'))
        WHERE metadata ? 'sha256'
    """)


def upgrade() -> None:
    # PG-only by design: the engram raw layer is PG-native-only by contract
    # (the endpoint rejects text_search_extension != "native"), so no Oracle
    # deployment ever carries rows with these metadata keys.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
