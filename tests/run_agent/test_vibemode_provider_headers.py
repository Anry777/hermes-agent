"""VibeMode provider-profile headers must reach every API surface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_build_anthropic_client_merges_provider_default_headers(monkeypatch):
    """Provider defaults such as VibeMode's WAF-safe UA must survive on Messages."""
    from agent import anthropic_adapter

    captured = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeAnthropicSdk:
        Anthropic = _FakeAnthropic

    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", _FakeAnthropicSdk)
    monkeypatch.setattr(anthropic_adapter, "normalize_proxy_env_vars", lambda: None)

    anthropic_adapter.build_anthropic_client(
        "sk-vibemode",
        "https://api.vibemod.pro/v1",
        default_headers={"User-Agent": "HermesAgent/1.0"},
    )

    assert captured["auth_token"] == "sk-vibemode"
    assert "api_key" not in captured
    assert captured["default_headers"]["User-Agent"] == "HermesAgent/1.0"


def test_build_anthropic_client_preserves_anthropic_beta_when_merging_headers(monkeypatch):
    """Merging provider headers must not clobber Anthropic beta headers."""
    from agent import anthropic_adapter

    captured = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeAnthropicSdk:
        Anthropic = _FakeAnthropic

    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", _FakeAnthropicSdk)
    monkeypatch.setattr(anthropic_adapter, "normalize_proxy_env_vars", lambda: None)
    monkeypatch.setattr(
        anthropic_adapter,
        "_common_betas_for_base_url",
        lambda *_args, **_kwargs: ["fine-grained-tool-streaming-2025-05-14"],
    )

    anthropic_adapter.build_anthropic_client(
        "sk-vibemode",
        "https://api.vibemod.pro/v1",
        default_headers={"User-Agent": "HermesAgent/1.0"},
    )

    assert captured["default_headers"]["User-Agent"] == "HermesAgent/1.0"
    assert captured["default_headers"]["anthropic-beta"] == "fine-grained-tool-streaming-2025-05-14"


def test_vibemode_anthropic_mode_agent_init_passes_provider_headers():
    """qwen3.7-max uses anthropic_messages, so init must pass VibeMode UA."""
    captured = {}

    def _fake_build_anthropic_client(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("agent.anthropic_adapter.build_anthropic_client", side_effect=_fake_build_anthropic_client),
    ):
        from run_agent import AIAgent

        AIAgent(
            api_key="sk-vibemode",
            base_url="https://api.vibemod.pro/v1",
            provider="vibemode",
            api_mode="anthropic_messages",
            model="qwen3.7-max",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert captured["default_headers"] == {"User-Agent": "HermesAgent/1.0"}


def test_vibemode_cloudflare_403_does_not_exhaust_credential_pool():
    """WAF 403 is a transport/header problem, not evidence that the key is dead."""
    from agent.error_classifier import FailoverReason
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="sk-vibemode",
        base_url="https://api.vibemod.pro/v1",
        provider="vibemode",
        api_mode="chat_completions",
        model="qwen3.7-max",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    calls = {"refresh": 0, "rotate": 0}

    class _FakePool:
        provider = "vibemode"

        def try_refresh_current(self):
            calls["refresh"] += 1
            return None

        def mark_exhausted_and_rotate(self, **_kwargs):
            calls["rotate"] += 1
            return None

        def has_available(self):
            return True

    agent._credential_pool = _FakePool()

    recovered, retried_429 = agent._recover_with_credential_pool(
        status_code=403,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context={"message": "Your request was blocked.", "reason": "error code: 1010"},
    )

    assert recovered is False
    assert retried_429 is False
    assert calls == {"refresh": 0, "rotate": 0}


def _build_openai_client_for_header_probe(provider: str):
    """Build the main-agent client and return its owned HTTPX transport."""
    from types import SimpleNamespace

    import httpx

    from agent.agent_runtime_helpers import create_openai_client

    transport = httpx.Client()
    captured = {}

    def _fake_openai(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    agent = SimpleNamespace(
        provider=provider,
        _build_keepalive_http_client=lambda *_args, **_kwargs: transport,
        _client_log_context=lambda: f"provider={provider}",
    )
    with patch("run_agent.OpenAI", side_effect=_fake_openai):
        create_openai_client(
            agent,
            {
                "api_key": "sk-redacted",
                "base_url": "https://api.vibemod.pro/v1",
                "default_headers": {"User-Agent": "HermesAgent/1.0"},
            },
            reason="test",
            shared=False,
        )

    assert captured["http_client"] is transport
    return transport


def test_vibemode_main_client_strips_openai_sdk_metadata_headers():
    """VibeMode GLM must not receive X-Stainless metadata added by the SDK."""
    import httpx

    transport = _build_openai_client_for_header_probe("vibemode")
    try:
        request = httpx.Request(
            "POST",
            "https://api.vibemod.pro/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-redacted",
                "User-Agent": "HermesAgent/1.0",
                "X-Stainless-Lang": "python",
                "X-Stainless-Package-Version": "2.24.0",
            },
        )
        for hook in transport.event_hooks.get("request", []):
            hook(request)

        assert "x-stainless-lang" not in request.headers
        assert "x-stainless-package-version" not in request.headers
        assert request.headers["authorization"] == "Bearer sk-redacted"
        assert request.headers["user-agent"] == "HermesAgent/1.0"
    finally:
        transport.close()


def test_non_vibemode_main_client_preserves_openai_sdk_metadata_headers():
    """The transport workaround must not alter requests for other providers."""
    import httpx

    transport = _build_openai_client_for_header_probe("openrouter")
    try:
        request = httpx.Request(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"X-Stainless-Lang": "python"},
        )
        for hook in transport.event_hooks.get("request", []):
            hook(request)

        assert request.headers["x-stainless-lang"] == "python"
    finally:
        transport.close()


def test_vibemode_aux_client_strips_openai_sdk_metadata_headers():
    """Auxiliary OpenAI clients must honor the same provider transport quirk."""
    import httpx

    from agent.auxiliary_client import _openai_http_client_kwargs

    kwargs = _openai_http_client_kwargs(
        "https://api.vibemod.pro/v1",
        provider_id="vibemode",
    )
    transport = kwargs["http_client"]
    try:
        request = httpx.Request(
            "POST",
            "https://api.vibemod.pro/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-redacted",
                "X-Stainless-Runtime": "CPython",
            },
        )
        for hook in transport.event_hooks.get("request", []):
            hook(request)

        assert "x-stainless-runtime" not in request.headers
        assert request.headers["authorization"] == "Bearer sk-redacted"
    finally:
        transport.close()


def test_vibemode_direct_aux_resolver_passes_provider_identity():
    """The real direct-provider resolver must carry VibeMode into HTTP setup."""
    from types import SimpleNamespace

    from agent import auxiliary_client

    captured = {}
    fake_client = SimpleNamespace(
        api_key="sk-redacted",
        base_url="https://api.vibemod.pro/v1",
    )

    def _fake_create_openai_client(**kwargs):
        captured.update(kwargs)
        return fake_client

    with (
        patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "api_key": "sk-redacted",
                "base_url": "https://api.vibemod.pro/v1",
            },
        ),
        patch.object(
            auxiliary_client,
            "_create_openai_client",
            side_effect=_fake_create_openai_client,
        ),
    ):
        client, model = auxiliary_client.resolve_provider_client(
            "vibemode",
            "glm-5.2",
        )

    assert client is fake_client
    assert model == "glm-5.2"
    assert captured["_provider_id"] == "vibemode"


def test_vibemode_async_conversion_installs_async_stainless_filter():
    """The real sync-to-async conversion must install an awaitable filter."""
    import asyncio
    from types import SimpleNamespace

    import httpx

    from agent.auxiliary_client import _to_async_client

    captured = {}
    sync_client = SimpleNamespace(
        api_key="sk-redacted",
        base_url="https://api.vibemod.pro/v1",
    )

    def _fake_async_openai(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch("openai.AsyncOpenAI", side_effect=_fake_async_openai):
        _to_async_client(sync_client, "glm-5.2")

    transport = captured["http_client"]
    try:
        request = httpx.Request(
            "POST",
            "https://api.vibemod.pro/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-redacted",
                "X-Stainless-Async": "async:asyncio",
            },
        )
        hooks = transport.event_hooks.get("request", [])
        assert hooks
        for hook in hooks:
            asyncio.run(hook(request))

        assert "x-stainless-async" not in request.headers
        assert request.headers["authorization"] == "Bearer sk-redacted"
    finally:
        asyncio.run(transport.aclose())
