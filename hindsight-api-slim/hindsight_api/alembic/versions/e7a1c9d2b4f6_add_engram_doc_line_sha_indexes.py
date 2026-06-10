"""Add expression indexes for engram raw units: (document_id, line_index) and (document_id, sha256)

Revision ID: e7a1c9d2b4f6
Revises: b2d4f6a8c1e3
Create Date: 2026-06-10

Engram raw units carry their transcript position in metadata->>'line_index'
(a stringified int — JSONB metadata is Dict[str, str] by contract). Two reads
need it indexed:

  1. Sequence reconstruction (the document units range endpoint):
       WHERE document_id = $1 AND metadata ? 'line_index'
         AND (metadata->>'line_index')::int BETWEEN $2 AND $3
       ORDER BY (metadata->>'line_index')::int
     A lexicographic sort on the raw string would break past line 9
     ("10" < "9"), so the index is built on the ::int expression.

  2. Idempotent re-send dedupe in the raw retain endpoint:
       NOT EXISTS (... document_id AND metadata->>'sha256' = ... )

Both are partial (engram rows only — rows that actually carry the key), so
they cost nothing on LLM-extracted banks.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e7a1c9d2b4f6"
down_revision: str | Sequence[str] | None = "b2d4f6a8c1e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_engram_doc_line
        ON {schema}memory_units (document_id, ((metadata->>'line_index')::int))
        WHERE metadata ? 'line_index'
    """)
    op.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_engram_doc_sha
        ON {schema}memory_units (document_id, (metadata->>'sha256'))
        WHERE metadata ? 'sha256'
    """)


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_engram_doc_line")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_engram_doc_sha")


def upgrade() -> None:
    # PG-only by design: the engram raw layer is PG-native-only by contract
    # (the endpoint rejects text_search_extension != "native"), so no Oracle
    # deployment ever carries rows with these metadata keys.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
