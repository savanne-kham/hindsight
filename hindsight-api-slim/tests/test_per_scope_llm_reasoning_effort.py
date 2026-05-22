"""Tests for per-scope ``HINDSIGHT_API_<SCOPE>_LLM_REASONING_EFFORT`` env vars.

Verifies:
- Global ``HINDSIGHT_API_LLM_REASONING_EFFORT`` defaults to "low" when unset.
- Each scope's env var parses as a string into the Config dataclass.
- Per-scope value overrides the global when set.
- Unset per-scope value falls back to the global at engine-init time (not parse time).
- Engine wires the resolved effort into each LLMConfig correctly.
"""

import os

import pytest


@pytest.fixture
def reset_env_reasoning_effort():
    """Snapshot + restore all reasoning_effort env vars around each test."""
    from hindsight_api.config import clear_config_cache

    keys = [
        "HINDSIGHT_API_LLM_REASONING_EFFORT",
        "HINDSIGHT_API_RETAIN_LLM_REASONING_EFFORT",
        "HINDSIGHT_API_REFLECT_LLM_REASONING_EFFORT",
        "HINDSIGHT_API_CONSOLIDATION_LLM_REASONING_EFFORT",
        # required to make get_config() succeed without a real LLM provider
        "HINDSIGHT_API_SKIP_LLM_VERIFICATION",
        "HINDSIGHT_API_LAZY_RERANKER",
        "HINDSIGHT_API_LLM_PROVIDER",
        "HINDSIGHT_API_LLM_MODEL",
    ]
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["HINDSIGHT_API_SKIP_LLM_VERIFICATION"] = "true"
    os.environ["HINDSIGHT_API_LAZY_RERANKER"] = "true"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_LLM_MODEL"] = "default-model"
    # Per-scope vars start unset; tests opt-in by setting them explicitly.
    for k in [
        "HINDSIGHT_API_LLM_REASONING_EFFORT",
        "HINDSIGHT_API_RETAIN_LLM_REASONING_EFFORT",
        "HINDSIGHT_API_REFLECT_LLM_REASONING_EFFORT",
        "HINDSIGHT_API_CONSOLIDATION_LLM_REASONING_EFFORT",
    ]:
        os.environ.pop(k, None)
    clear_config_cache()
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    clear_config_cache()


class TestPerScopeLLMReasoningEffort:
    def test_global_defaults_to_low_when_unset(self, reset_env_reasoning_effort):
        from hindsight_api.config import get_config

        config = get_config()
        assert config.llm_reasoning_effort == "low"
        assert config.retain_llm_reasoning_effort is None
        assert config.reflect_llm_reasoning_effort is None
        assert config.consolidation_llm_reasoning_effort is None

    def test_per_scope_env_parses_as_string(self, reset_env_reasoning_effort):
        from hindsight_api.config import clear_config_cache, get_config

        os.environ["HINDSIGHT_API_RETAIN_LLM_REASONING_EFFORT"] = "low"
        os.environ["HINDSIGHT_API_REFLECT_LLM_REASONING_EFFORT"] = "high"
        os.environ["HINDSIGHT_API_CONSOLIDATION_LLM_REASONING_EFFORT"] = "medium"
        clear_config_cache()
        config = get_config()
        assert config.retain_llm_reasoning_effort == "low"
        assert config.reflect_llm_reasoning_effort == "high"
        assert config.consolidation_llm_reasoning_effort == "medium"

    def test_global_only_leaves_per_scope_none(self, reset_env_reasoning_effort):
        from hindsight_api.config import clear_config_cache, get_config

        os.environ["HINDSIGHT_API_LLM_REASONING_EFFORT"] = "high"
        clear_config_cache()
        config = get_config()
        assert config.llm_reasoning_effort == "high"
        # Per-scope fields stay None — fallback happens at engine init, not parse time.
        assert config.retain_llm_reasoning_effort is None
        assert config.reflect_llm_reasoning_effort is None
        assert config.consolidation_llm_reasoning_effort is None

    def test_per_scope_overrides_global_at_engine_init(self, reset_env_reasoning_effort):
        from hindsight_api.config import clear_config_cache, get_config
        from hindsight_api import MemoryEngine

        os.environ["HINDSIGHT_API_LLM_REASONING_EFFORT"] = "low"
        os.environ["HINDSIGHT_API_REFLECT_LLM_REASONING_EFFORT"] = "high"
        clear_config_cache()

        engine = MemoryEngine(
            memory_llm_provider="mock",
            memory_llm_model="default-model",
            skip_llm_verification=True,
            lazy_reranker=True,
        )

        # retain falls back to global "low"
        assert engine._retain_llm_config.reasoning_effort == "low"
        # reflect uses its per-scope override "high"
        assert engine._reflect_llm_config.reasoning_effort == "high"
        # consolidation falls back to global "low"
        assert engine._consolidation_llm_config.reasoning_effort == "low"
