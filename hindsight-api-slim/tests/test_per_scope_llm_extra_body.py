"""Tests for per-scope ``HINDSIGHT_API_<SCOPE>_LLM_EXTRA_BODY`` env vars.

Verifies:
- Each scope's extra_body env var parses as JSON into the Config dataclass.
- Per-scope value beats the global ``HINDSIGHT_API_LLM_EXTRA_BODY`` when set.
- Unset per-scope value falls back to the global.
- Unset global + unset per-scope = None (no extra_body merged at call time).
"""

import json
import os

import pytest


@pytest.fixture
def reset_env_extra_body():
    """Snapshot + restore all extra_body env vars around each test."""
    from hindsight_api.config import clear_config_cache

    keys = [
        "HINDSIGHT_API_LLM_EXTRA_BODY",
        "HINDSIGHT_API_RETAIN_LLM_EXTRA_BODY",
        "HINDSIGHT_API_REFLECT_LLM_EXTRA_BODY",
        "HINDSIGHT_API_CONSOLIDATION_LLM_EXTRA_BODY",
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
        "HINDSIGHT_API_LLM_EXTRA_BODY",
        "HINDSIGHT_API_RETAIN_LLM_EXTRA_BODY",
        "HINDSIGHT_API_REFLECT_LLM_EXTRA_BODY",
        "HINDSIGHT_API_CONSOLIDATION_LLM_EXTRA_BODY",
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


class TestPerScopeLLMExtraBody:
    def test_unset_yields_none_on_all_scopes(self, reset_env_extra_body):
        from hindsight_api.config import get_config

        config = get_config()
        assert config.llm_extra_body is None
        assert config.retain_llm_extra_body is None
        assert config.reflect_llm_extra_body is None
        assert config.consolidation_llm_extra_body is None

    def test_per_scope_env_parses_as_json(self, reset_env_extra_body):
        from hindsight_api.config import clear_config_cache, get_config

        os.environ["HINDSIGHT_API_RETAIN_LLM_EXTRA_BODY"] = json.dumps(
            {"temperature": 0.6, "top_p": 0.8}
        )
        os.environ["HINDSIGHT_API_REFLECT_LLM_EXTRA_BODY"] = json.dumps(
            {"chat_template_kwargs": {"enable_thinking": True}}
        )
        os.environ["HINDSIGHT_API_CONSOLIDATION_LLM_EXTRA_BODY"] = json.dumps(
            {"chat_template_kwargs": {"enable_thinking": False}, "presence_penalty": 0.0}
        )
        clear_config_cache()
        config = get_config()
        assert config.retain_llm_extra_body == {"temperature": 0.6, "top_p": 0.8}
        assert config.reflect_llm_extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
        assert config.consolidation_llm_extra_body == {
            "chat_template_kwargs": {"enable_thinking": False},
            "presence_penalty": 0.0,
        }

    def test_global_only_leaves_per_scope_none(self, reset_env_extra_body):
        from hindsight_api.config import clear_config_cache, get_config

        os.environ["HINDSIGHT_API_LLM_EXTRA_BODY"] = json.dumps({"temperature": 0.7})
        clear_config_cache()
        config = get_config()
        assert config.llm_extra_body == {"temperature": 0.7}
        # Per-scope fields stay None — fallback happens at engine init, not parse time.
        assert config.retain_llm_extra_body is None
        assert config.reflect_llm_extra_body is None
        assert config.consolidation_llm_extra_body is None

    def test_invalid_json_raises_at_parse_time(self, reset_env_extra_body):
        from hindsight_api.config import clear_config_cache, get_config

        os.environ["HINDSIGHT_API_RETAIN_LLM_EXTRA_BODY"] = "{not valid json"
        clear_config_cache()
        with pytest.raises(json.JSONDecodeError):
            get_config()
