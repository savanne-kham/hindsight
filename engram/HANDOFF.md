# engram — agent handoff context

> Paste this document to any new agent (or tell it to read
> `~/dev/ai-lab/hindsight/engram/engram/HANDOFF.md`) to continue improving the
> engram memory system. Last updated: 2026-06-10.

## Mission

Self-owned, audit-grade, long-term memory for the owner's AI agents (Claude Code,
codex, hermes-agent) built on a **fork** of Hindsight. Two layers:

1. **Raw capture (`engram-raw` bank)** — every agent-session message stored
   VERBATIM + every tool action as a deterministic `cc-style/v1` rendering.
   **Zero LLM in the write path** (~100 ms warm). Immutable, sha256-sealed:
   non-repudiation. The session JSONL files remain the byte-exact forensic archive.
2. **Derived projections** (future "normalization layer") — facts, observations,
   per-turn groupings, distilled units. Always REGENERABLE offline from raw.
   Never enrich the raw log; project it (event sourcing / CQRS).

## Owner philosophy (read this first)

**Nothing here is locked.** The owner wants this system continuously questioned
and improved — aim for SOTA and big ROI, learn from how top teams and top
developers solve the same problems. **No assumptions: reason from empirical
facts** (measure, reproduce, read the actual spec/source — every major decision
below was settled by querying the live DB, sampling live stacks, or running a
falsifiable test, not by plausibility). If you think a design choice below is
wrong, challenge it — *with evidence*.

One exception, non-negotiable:
- **NO upstream PRs to vectorize-io/hindsight.** Ever. All fixes stay fork-only
  on the `engram` branch (owner dislikes the maintainers).

## Current design decisions (challenge with evidence, not vibes)

- Fork stays rebase-friendly: new files only + minimal upstream deltas
  (`git rebase upstream/main` is the update path; the engram feature is
  `hindsight_api/api/engram_raw.py` + a 3-line hook in `http.py` + migrations).
- Raw layer stays atomic: 1 message ↔ 1 unit for human/assistant text. Grouping
  happens at READ time (`turn` key, `?around` endpoint) or in derived layers.
- Conversation language follows the owner (French); code/comments/docs English.
- Conventional commits, one-liners, no Co-Authored-By.

## System map

| Piece | Where |
|---|---|
| Fork worktree (runtime serves THIS) | `~/dev/ai-lab/hindsight/engram` (branch `engram`) — online: <https://github.com/savanne-kham/hindsight/tree/engram> (`origin`; `upstream`=vectorize-io/hindsight) |
| API runtime | launchd `com.guinsoo.hindsight` on **stormwind**, port 8888 (`0.0.0.0` — M4 reaches it via Tailscale `http://stormwind:8888`); venv `~/dev/ai-lab/hindsight-config/macstudio-m1max/venv-hindsight` (editable install → the worktree); start script in `~/dev/ai-lab/hindsight-config/` |
| Write hook (Claude Code) | `~/.claude/hooks/hindsight-retain.py` → symlink into `~/.dotfiles/common/claude/.claude/hooks/` (dotfiles repo, GitLab). Events Stop/SessionEnd/SessionStart, merged into settings.json by `install.sh` |
| Reconcile sweep | launchd `com.guinsoo.engram-raw-sweep` every 3 h + SessionStart (plists versioned per profile in dotfiles `<profile>/launchd/`) |
| DB | PostgreSQL 17 local, db `hindsight`, table `memory_units` (metadata jsonb, `Dict[str,str]` ONLY) |
| Models | bge-m3 embeddings + bge-reranker-v2-m3, local on MPS; LLMs = local llama.cpp (35B :11434, 30B retain :11435), NOT used by the raw path |
| Contract doc | `engram/README.md` (this dir) — schema `engram-raw/v1`, endpoints, test invocation |

## Key mechanisms (all implemented & tested, 2026-06-10)

- **Cursor protocol (hook)**: `<sid>.pos` = committed watermark, advanced ONLY on
  server 2xx; `<sid>.pending` = in-flight reservation; failed slices re-sent by
  next flush/sweep. `<sid>.last` = persisted digest of last text message
  (cross-slice consecutive dedupe).
- **Sequence**: order by `(document_id, (metadata->>'line_index')::int)` — sort
  key, NOT a linked list. The `::int` cast is mandatory ("10" < "9" as strings).
  Partial expression indexes via migration `e7a1c9d2b4f6`.
  Read API: `GET /v1/engram/banks/{bank}/documents/{doc}/units?around=N&radius=k`
  (small-to-big retrieval).
- **Idempotent re-sends**: server skips `(document_id, line_index, sha256)`
  duplicates, reports them as `duplicates`. CLIENT CONTRACT:
  `count + duplicates > 0` = success (count-only loops forever).
- **Unit shape**: `turn` metadata = line_index of initiating user message
  (grouping key); runs of sparse tool actions (no result, no error, <300 chars)
  merged into one `tool-batch` unit; long tool results truncated head+tail
  (errors live at the tail); `error` tag from harness `is_error` flag (no
  substring sniffing); result-bearing/diff-rich actions stay atomic.
- **MPS safety**: every transformer entry point must bound sequence length.
  Embeddings: `EMBED_CHAR_CAP=6000` + sub-batch 8. Reranker:
  `HINDSIGHT_API_RERANKER_LOCAL_MAX_LENGTH=1024` (also the model's training
  regime). Unbounded → O(n²) attention → Metal wedges silently
  (`waitUntilCompleted` forever) and ALL later MPS ops in the process hang.

## How to work on this

```bash
cd ~/dev/ai-lab/hindsight/engram/hindsight-api-slim
uv run ruff check . && uv run ruff format . && uv run ty check hindsight_api/   # lint (mandatory)
# tests for the engram endpoints (NOT the embedded db):
psql -d postgres -c "CREATE DATABASE hindsight_test OWNER hindsight" 2>/dev/null
psql -d hindsight_test -c "CREATE EXTENSION IF NOT EXISTS vector"   # needs superuser
HINDSIGHT_API_DATABASE_URL=postgresql://hindsight:hindsight@127.0.0.1:5432/hindsight_test \
  uv run --extra local-ml pytest tests/test_engram_raw.py -n 0     # -n 0: xdist races migrations
# deploy = restart the service (runtime is editable on this worktree, auto-migrates):
launchctl kickstart -k gui/$(id -u)/com.guinsoo.hindsight
# do NOT send requests in the first ~30 s after boot (MPS model-init race)
```

Debugging gold: `sample <pid> 3` (native macOS, no root) gives live stacks —
it's how the MPS wedge was found. Logs: `~/logs/hindsight/hindsight.log`,
hook log `~/.claude/hindsight-retain.log` (`✓`=committed, `✗`=will retry, `+N dup`).

## Deep documentation & related repos

Obsidian vault `~/obsidian-vault/03 - AI/memory/` — the owner's whole body of
**agentic-memory notes**: model benchmarks (retain/reflect/consolidate), provider
comparisons, architecture and Postgres data model, consolidation strategy…
Browse it before redesigning anything. Key notes in `memory/hindsight/`:

- `engram-raw-sequence-and-idempotence.md` — design decisions + rationale
- `mps-attention-wedge-reranker.md` — the MPS wedge story + **MaxP windowing
  parked in its Future-work section** (do NOT raise max_length instead)
- `claude-code-retain-hook.md`, `hindsight-architecture.md`,
  `hindsight-postgres-data-model.md` — older foundations
- Session log: vault `90 - AI Sessions/hindsight.md` (read last entry first)

Related repo: <https://gitlab.com/savanne-kham/ai-agent-sessions> — **draft**
exporter of agentic sessions (Claude Code, codex, opencode); upstream feeder /
companion of this capture layer, same cross-agent memory ambition.

## Backlog (next steps, in rough priority order)

1. **Normalization layer** — the big one: offline, deterministic per-`turn`
   projection of raw units (narration + actions grouped), separate bank or
   `derived` tag; later LLM distillation on top. Regenerable, never mutates raw.
2. **Wire `?around` into recall** — a recall hit on a raw unit should expand to
   its conversational neighborhood (MCP/hermes side).
3. **Purge historical duplicates** — pre-fix 210/216-style rows still in DB
   (same text, different line_index, no other text between); one-shot SQL or
   let normalization handle them.
4. **MaxP windowing** for long-unit reranking — ONLY if quality on raw units
   proves critical AND normalization doesn't make it moot (see vault note).
5. **M4 sync** — on BA4714: `git pull && ./install.sh` in dotfiles + symlink &
   `launchctl bootstrap` the `engram-raw-sweep` plist (README has the commands).

## State as of 2026-06-10

`engram` branch = upstream/main (20 commits integrated, clean rebase) + 9 fork
commits, pushed to `origin/engram`. Runtime live on this exact code, verified:
raw POST ~0.3-0.5 s, worst-case recall 16 s (was: wedged the server), dedupe
idempotent E2E, 3 pytest green. Dotfiles side committed (hook v2), on GitLab
`ahead` — push pending owner decision.
