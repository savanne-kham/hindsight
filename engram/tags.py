"""Deterministic tagging for engram raw memory units.

Pure functions, zero LLM, zero I/O — trivially portable to Rust later.
Detection is case-insensitive substring matching, adapted from the V1
taxonomy in ai-agent-sessions/scripts/export_codex_sessions.py.
"""

BASE_TAGS = ["raw", "agent-session"]

# tag -> case-insensitive substrings that trigger it
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


def dedupe_preserve_order(tags):
    seen = set()
    out = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def detect_topic_tags(text):
    haystack = text.lower()
    return [
        tag
        for tag, keywords in TOPIC_KEYWORDS.items()
        if any(keyword in haystack for keyword in keywords)
    ]


def build_tags(*, source, host, project, role, text):
    tags = [
        *BASE_TAGS,
        source,
        host,
        f"project:{project}",
        f"role:{role}",
        *detect_topic_tags(text),
    ]
    return dedupe_preserve_order(tags)
