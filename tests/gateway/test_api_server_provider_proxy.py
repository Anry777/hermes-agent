"""Tests for API Server provider_proxy mode."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.api_server_provider_proxy import APIServerProviderProxy, ResolvedProxyTarget
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware


def test_codex_client_includes_cloudflare_required_headers():
    target = ResolvedProxyTarget(
        public_id="gpt-5.6-luna",
        provider="openai-codex",
        model="gpt-5.6-luna",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="test-token",
        runtime={},
    )
    required_headers = {
        "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
        "originator": "codex_cli_rs",
    }

    with patch("gateway.api_server_provider_proxy.OpenAI") as openai_cls, patch(
        "agent.auxiliary_client._codex_cloudflare_headers",
        return_value=required_headers,
    ) as build_headers:
        client = APIServerProviderProxy._create_openai_client(target)

    assert client is openai_cls.return_value
    build_headers.assert_called_once_with("test-token")
    openai_cls.assert_called_once_with(
        api_key="test-token",
        base_url="https://chatgpt.com/backend-api/codex",
        default_headers=required_headers,
    )


def test_vibemode_proxy_client_uses_profile_headers_and_request_filter():
    import httpx

    target = ResolvedProxyTarget(
        public_id="glm-5.2",
        provider="vibemode",
        model="glm-5.2",
        api_mode="chat_completions",
        base_url="https://api.vibemod.pro/v1",
        api_key="sk-redacted",
        runtime={},
    )
    captured = {}

    def _fake_openai(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch(
        "gateway.api_server_provider_proxy.OpenAI",
        side_effect=_fake_openai,
    ):
        APIServerProviderProxy._create_openai_client(target)

    assert captured["default_headers"] == {"User-Agent": "HermesAgent/1.0"}
    transport = captured["http_client"]
    try:
        request = httpx.Request(
            "POST",
            "https://api.vibemod.pro/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-redacted",
                "X-Stainless-Lang": "python",
            },
        )
        for hook in transport.event_hooks.get("request", []):
            hook(request)

        assert "x-stainless-lang" not in request.headers
        assert request.headers["authorization"] == "Bearer sk-redacted"
    finally:
        transport.close()


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/models", adapter._handle_models)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    return app


def _proxy_adapter() -> APIServerAdapter:
    return APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "mode": "provider_proxy",
                "provider_proxy": {
                    "allow_streaming": True,
                    "models": [
                        {
                            "id": "openrouter/test-model",
                            "provider": "openrouter",
                            "model": "upstream/test-model",
                        },
                        {
                            "id": "gpt-5.4",
                            "provider": "openai-codex",
                            "model": "gpt-5.4",
                        },
                    ]
                },
            },
        )
    )


def _codex_responses_proxy_adapter() -> APIServerAdapter:
    return APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "mode": "codex_responses_proxy",
                "codex_responses_proxy": {
                    "default_model": "gpt-5.4",
                    "model_discovery": "live",
                    "allow_models": ["^gpt-5\\."],
                    "deny_models": ["mini$"],
                },
            },
        )
    )


class _MockChatResponse:
    def model_dump(self):
        return {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": 123,
            "model": "upstream/test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "proxied"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }


class _MockStreamChunk:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


class _UpstreamStatusError(Exception):
    def __init__(self, status_code, message="upstream failed"):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code)


def _read_file_tool():
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }


def _sse_data_payloads(raw: str):
    payloads = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            payloads.append("[DONE]")
        else:
            payloads.append(data)
    return payloads


def _sse_events(raw: str):
    events = []
    event_name = None
    for line in raw.splitlines():
        if line.startswith("event: "):
            event_name = line[len("event: "):]
        elif line.startswith("data: "):
            data = line[len("data: "):]
            events.append((event_name, data))
            event_name = None
    return events


class TestProviderProxyModels:
    @pytest.mark.asyncio
    async def test_models_returns_proxy_catalog(self):
        adapter = _proxy_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()

        assert data["object"] == "list"
        assert [item["id"] for item in data["data"]] == [
            "openrouter/test-model",
            "gpt-5.4",
        ]
        assert data["data"][0]["owned_by"] == "openrouter"
        assert data["data"][1]["owned_by"] == "openai-codex"


class TestCodexResponsesProxy:
    @pytest.mark.asyncio
    async def test_models_uses_live_codex_discovery_with_filters(self):
        adapter = _codex_responses_proxy_adapter()
        app = _create_app(adapter)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"slug": "gpt-5.4", "object": "model"},
                {"slug": "gpt-5.4-mini", "object": "model"},
                {"slug": "codex-internal", "object": "model"},
            ]
        }

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.httpx.get", return_value=mock_response) as mock_get:
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "***",
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/models")
                data = await resp.json()

        assert resp.status == 200
        assert adapter._mode == "codex_responses_proxy"
        assert [item["id"] for item in data["data"]] == ["gpt-5.4"]
        mock_response.raise_for_status.assert_called_once_with()
        mock_get.assert_called_once_with(
            "https://chatgpt.com/backend-api/codex/models",
            params={"client_version": "1.0.0"},
            headers={"Authorization": "Bearer ***"},
            timeout=15,
        )
        mock_resolve.assert_called_once_with(
            requested="openai-codex",
            explicit_base_url=None,
            explicit_api_key=None,
            target_model="gpt-5.4",
        )

    @pytest.mark.asyncio
    async def test_responses_proxy_bypasses_agent_and_passes_responses_body(self):
        adapter = _codex_responses_proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        mock_client = MagicMock()
        mock_client.responses.create.return_value = SimpleNamespace(
            id="resp_123",
            object="response",
            model="gpt-5.4",
            output=[],
        )

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://api.openai.com/v1",
                "api_key": "***",
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "gpt-5.4",
                        "input": "hello",
                        "model_reasoning_effort": "high",
                        "temperature": 0.2,
                        "client_metadata": {"client": "vscode-codex"},
                    },
                )
                data = await resp.json()

        assert resp.status == 200
        assert data["id"] == "resp_123"
        adapter._run_agent.assert_not_called()
        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["model"] == "gpt-5.4"
        assert kwargs["input"] == "hello"
        assert kwargs["stream"] is False
        assert kwargs["reasoning"] == {"effort": "high"}
        assert "model_reasoning_effort" not in kwargs
        assert "client_metadata" not in kwargs
        assert kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_responses_proxy_records_pool_success_for_selected_credential(self):
        adapter = _codex_responses_proxy_adapter()
        app = _create_app(adapter)

        pool = MagicMock()
        pool.current.return_value = SimpleNamespace(id="cred-1", label="Primary")
        mock_client = MagicMock()
        mock_client.responses.create.return_value = SimpleNamespace(
            id="resp_123",
            object="response",
            model="gpt-5.4",
            output=[],
        )

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://api.openai.com/v1",
                "api_key": "***",
                "credential_pool": pool,
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "gpt-5.4", "input": "hello"},
                )
                data = await resp.json()

        assert resp.status == 200
        assert data["id"] == "resp_123"
        pool.record_success.assert_called_once_with("cred-1")
        pool.mark_exhausted_and_rotate.assert_not_called()

    @pytest.mark.asyncio
    async def test_responses_proxy_marks_selected_pool_credential_exhausted_on_rate_limit(self):
        adapter = _codex_responses_proxy_adapter()
        app = _create_app(adapter)

        pool = MagicMock()
        pool.current.return_value = SimpleNamespace(id="cred-1", label="Primary")
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = _UpstreamStatusError(429, "rate limited")

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-selected",
                "credential_pool": pool,
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "gpt-5.4", "input": "hello"},
                )
                data = await resp.json()

        assert resp.status == 502
        assert data["error"]["code"] == "codex_responses_proxy_error"
        pool.mark_exhausted_and_rotate.assert_called_once_with(
            status_code=429,
            error_context={"message": "rate limited"},
            credential_id="cred-1",
            api_key_hint="sk-selected",
        )
        pool.record_success.assert_not_called()

    @pytest.mark.asyncio
    async def test_responses_proxy_stream_records_pool_success_after_complete_stream(self):
        adapter = _codex_responses_proxy_adapter()
        app = _create_app(adapter)

        pool = MagicMock()
        pool.current.return_value = SimpleNamespace(id="cred-1", label="Primary")
        stream_events = iter([
            SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_123", model="gpt-5.4")),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(id="resp_123", model="gpt-5.4")),
        ])
        mock_client = MagicMock()
        mock_client.responses.create.return_value = stream_events

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "***",
                "credential_pool": pool,
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "gpt-5.4", "input": "hello", "stream": True},
                )
                raw = await resp.text()

        assert resp.status == 200
        assert raw.rstrip().endswith("data: [DONE]")
        pool.record_success.assert_called_once_with("cred-1")
        pool.mark_exhausted_and_rotate.assert_not_called()

    @pytest.mark.asyncio
    async def test_responses_proxy_stream_marks_pool_credential_exhausted_when_open_fails(self):
        adapter = _codex_responses_proxy_adapter()
        app = _create_app(adapter)

        pool = MagicMock()
        pool.current.return_value = SimpleNamespace(id="cred-1", label="Primary")
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = _UpstreamStatusError(401, "token invalidated")

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "sk-selected",
                "credential_pool": pool,
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "gpt-5.4", "input": "hello", "stream": True},
                )
                await resp.text()

        pool.mark_exhausted_and_rotate.assert_called_once_with(
            status_code=401,
            error_context={"message": "token invalidated"},
            credential_id="cred-1",
            api_key_hint="sk-selected",
        )
        pool.record_success.assert_not_called()

    @pytest.mark.asyncio
    async def test_responses_proxy_streams_responses_sse_without_chat_conversion(self):
        adapter = _codex_responses_proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        stream_events = iter([
            SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_123", model="gpt-5.4")),
            SimpleNamespace(type="response.output_text.delta", delta="hello"),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(id="resp_123", model="gpt-5.4")),
        ])
        mock_client = MagicMock()
        mock_client.responses.create.return_value = stream_events

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "***",
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "gpt-5.4",
                        "input": "hello",
                        "stream": True,
                        "store": True,
                        "temperature": 0.2,
                        "max_output_tokens": 16,
                        "reasoning": {"effort": "low", "summary": "none"},
                        "client_metadata": {"client": "vscode-codex"},
                    },
                )
                raw = await resp.text()

        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        events = _sse_events(raw)
        assert events[-1] == (None, "[DONE]")
        assert [name for name, data in events[:-1]] == [
            "response.created",
            "response.output_text.delta",
            "response.completed",
        ]
        assert "chat.completion.chunk" not in raw
        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["store"] is False
        assert kwargs["instructions"] == "You are a helpful assistant."
        assert kwargs["input"] == [{"role": "user", "content": "hello"}]
        assert kwargs["reasoning"] == {"effort": "low"}
        assert "model_reasoning_effort" not in kwargs
        assert "max_output_tokens" not in kwargs
        assert "client_metadata" not in kwargs
        assert "temperature" not in kwargs
        adapter._run_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_completions_is_rejected_in_codex_responses_proxy_mode(self):
        adapter = _codex_responses_proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]},
            )
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["code"] == "unsupported_operation"
        adapter._run_agent.assert_not_called()


class TestProviderProxyChatCompletions:
    @pytest.mark.asyncio
    async def test_unknown_model_returns_model_not_found(self):
        adapter = _proxy_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "missing/model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            data = await resp.json()

        assert resp.status == 404
        assert data["error"]["code"] == "model_not_found"
        assert data["error"]["param"] == "model"

    @pytest.mark.asyncio
    async def test_chat_completions_proxy_bypasses_agent_and_uses_internal_model(self):
        adapter = _proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _MockChatResponse()

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client) as mock_openai:
            mock_resolve.return_value = {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "sk-test",
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openrouter/test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "temperature": 0.2,
                    },
                )
                data = await resp.json()

        assert resp.status == 200
        assert data["model"] == "openrouter/test-model"
        assert data["choices"][0]["message"]["content"] == "proxied"
        adapter._run_agent.assert_not_called()
        mock_resolve.assert_called_once_with(
            requested="openrouter",
            explicit_base_url=None,
            explicit_api_key=None,
            target_model="upstream/test-model",
        )
        mock_openai.assert_called_once_with(
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
        )
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "upstream/test-model"
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
        assert kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_codex_responses_provider_uses_chat_compat_adapter(self):
        adapter = _proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        mock_client = MagicMock()
        mock_response = object()
        mock_client.responses.create.return_value = mock_response
        mock_transport = MagicMock()
        mock_transport.normalize_response.return_value = SimpleNamespace(
            content="codex proxied",
            finish_reason="stop",
            tool_calls=None,
        )

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client), \
             patch("agent.transports.codex.ResponsesApiTransport", return_value=mock_transport):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-test",
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "messages": [
                            {"role": "system", "content": "be concise"},
                            {"role": "user", "content": "hello"},
                        ],
                        "max_completion_tokens": 32,
                    },
                )
                data = await resp.json()

        assert resp.status == 200
        assert data["model"] == "gpt-5.4"
        assert data["choices"][0]["message"]["content"] == "codex proxied"
        adapter._run_agent.assert_not_called()
        mock_resolve.assert_called_once_with(
            requested="openai-codex",
            explicit_base_url=None,
            explicit_api_key=None,
            target_model="gpt-5.4",
        )
        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["model"] == "gpt-5.4"
        assert kwargs["instructions"] == "be concise"
        assert kwargs["max_output_tokens"] == 32
        assert kwargs["store"] is False
        mock_transport.normalize_response.assert_called_once_with(mock_response)

    @pytest.mark.asyncio
    async def test_codex_responses_provider_returns_openai_tool_calls_and_payload_passthrough(self):
        adapter = _proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        mock_client = MagicMock()
        mock_response = object()
        mock_client.responses.create.return_value = mock_response
        mock_transport = MagicMock()
        mock_transport.normalize_response.return_value = SimpleNamespace(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                SimpleNamespace(
                    id="call_read",
                    function=SimpleNamespace(name="read_file", arguments='{"path":"README.md"}'),
                )
            ],
        )

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client), \
             patch("agent.transports.codex.ResponsesApiTransport", return_value=mock_transport):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://api.openai.com/v1",
                "api_key": "***",
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "read README"}],
                        "tools": [_read_file_tool()],
                        "tool_choice": {"type": "function", "function": {"name": "read_file"}},
                        "parallel_tool_calls": False,
                        "temperature": 0.1,
                    },
                )
                data = await resp.json()

        assert resp.status == 200
        message = data["choices"][0]["message"]
        assert message["content"] == ""
        assert message["tool_calls"] == [
            {
                "id": "call_read",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
            }
        ]
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["tools"] == [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a text file",
                "strict": False,
                "parameters": _read_file_tool()["function"]["parameters"],
            }
        ]
        assert kwargs["tool_choice"] == {"type": "function", "name": "read_file"}
        assert kwargs["parallel_tool_calls"] is False
        assert kwargs["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_chatgpt_codex_filters_roocode_sampling_params_and_maps_reasoning_effort(self):
        adapter = _proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        mock_client = MagicMock()
        mock_response = SimpleNamespace(output=[])
        mock_client.responses.create.return_value = mock_response
        mock_transport = MagicMock()
        mock_transport.normalize_response.return_value = SimpleNamespace(
            content="codex proxied",
            finish_reason="stop",
            tool_calls=None,
        )

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client), \
             patch("agent.transports.codex.ResponsesApiTransport", return_value=mock_transport):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "***",
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "messages": [{"role": "user", "content": "hello"}],
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "presence_penalty": 0.1,
                        "frequency_penalty": 0.1,
                        "seed": 123,
                        "logprobs": True,
                        "top_logprobs": 2,
                        "reasoning_effort": "high",
                    },
                )
                data = await resp.json()

        assert resp.status == 200
        assert data["model"] == "gpt-5.4"
        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["reasoning"] == {"effort": "high"}
        for unsupported_key in (
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "logprobs",
            "top_logprobs",
        ):
            assert unsupported_key not in kwargs

    @pytest.mark.asyncio
    async def test_chatgpt_codex_normalizes_minimal_reasoning_effort_for_streaming(self):
        adapter = _proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        terminal_response = SimpleNamespace(output=[], usage={"input_tokens": 4, "output_tokens": 2})
        stream_events = iter([
            SimpleNamespace(type="response.output_text.delta", delta="ok"),
            SimpleNamespace(type="response.completed", response=terminal_response),
        ])
        mock_client = MagicMock()
        mock_client.responses.create.return_value = stream_events
        mock_transport = MagicMock()
        mock_transport.normalize_response.return_value = SimpleNamespace(
            content="ok",
            finish_reason="stop",
            tool_calls=None,
        )

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client), \
             patch("agent.transports.codex.ResponsesApiTransport", return_value=mock_transport):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "***",
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hello"}],
                        "temperature": 0.2,
                        "reasoning_effort": "minimal",
                    },
                )
                raw = await resp.text()

        assert resp.status == 200
        assert _sse_data_payloads(raw)[-1] == "[DONE]"
        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["reasoning"] == {"effort": "low"}
        assert "temperature" not in kwargs

    @pytest.mark.asyncio
    async def test_codex_responses_multimodal_and_tool_result_roundtrip(self):
        adapter = _proxy_adapter()
        app = _create_app(adapter)

        mock_client = MagicMock()
        mock_client.responses.create.return_value = SimpleNamespace(output=[])
        mock_transport = MagicMock()
        mock_transport.normalize_response.return_value = SimpleNamespace(
            content="done",
            finish_reason="stop",
            tool_calls=None,
        )

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client), \
             patch("agent.transports.codex.ResponsesApiTransport", return_value=mock_transport):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://api.openai.com/v1",
                "api_key": "***",
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "what is in this image?"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"},
                                    },
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_read",
                                        "type": "function",
                                        "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                                    }
                                ],
                            },
                            {"role": "tool", "tool_call_id": "call_read", "content": "file contents"},
                        ],
                    },
                )

        assert resp.status == 200
        payload_input = mock_client.responses.create.call_args.kwargs["input"]
        assert payload_input[0] == {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is in this image?"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "low"},
            ],
        }
        assert {"type": "function_call", "call_id": "call_read", "name": "read_file", "arguments": '{"path":"README.md"}'} in payload_input
        assert {"type": "function_call_output", "call_id": "call_read", "output": "file contents"} in payload_input

    @pytest.mark.asyncio
    async def test_codex_responses_rejects_unsupported_file_content_parts(self):
        adapter = _proxy_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.4",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "read this"},
                                {"type": "input_file", "file_id": "file_123"},
                            ],
                        }
                    ],
                },
            )
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["code"] == "unsupported_content_type"
        assert data["error"]["param"] == "messages[0].content"

    @pytest.mark.asyncio
    async def test_responses_api_is_rejected_in_proxy_mode_without_agent(self):
        adapter = _proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/responses", json={"input": "hello"})
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["code"] == "unsupported_operation"
        adapter._run_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_api_is_rejected_in_proxy_mode_without_agent(self):
        adapter = _proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": "hello"})
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["code"] == "unsupported_operation"
        adapter._run_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_completions_streaming_proxies_sse_chunks(self):
        adapter = _proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([
            _MockStreamChunk({
                "id": "chatcmpl-upstream",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "upstream/test-model",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }),
            _MockStreamChunk({
                "id": "chatcmpl-upstream",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "upstream/test-model",
                "choices": [{"index": 0, "delta": {"content": "streamed"}, "finish_reason": None}],
            }),
            _MockStreamChunk({
                "id": "chatcmpl-upstream",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "upstream/test-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }),
        ])

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client):
            mock_resolve.return_value = {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "***",
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openrouter/test-model",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
                raw = await resp.text()

        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        payloads = _sse_data_payloads(raw)
        assert payloads[-1] == "[DONE]"
        chunks = [json.loads(item) for item in payloads if item != "[DONE]"]
        assert [chunk["model"] for chunk in chunks] == ["openrouter/test-model"] * 3
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[1]["choices"][0]["delta"] == {"content": "streamed"}
        assert chunks[2]["choices"][0]["finish_reason"] == "stop"
        adapter._run_agent.assert_not_called()
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "upstream/test-model"
        assert kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_codex_responses_streaming_is_adapted_to_chat_completion_sse(self):
        adapter = _proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        terminal_response = SimpleNamespace(output=[], usage={"input_tokens": 4, "output_tokens": 2})
        stream_events = iter([
            SimpleNamespace(type="response.output_text.delta", delta="codex "),
            SimpleNamespace(type="response.output_text.delta", delta="stream"),
            SimpleNamespace(type="response.completed", response=terminal_response),
        ])
        mock_client = MagicMock()
        mock_client.responses.create.return_value = stream_events
        mock_transport = MagicMock()
        mock_transport.normalize_response.return_value = SimpleNamespace(
            content="codex stream",
            finish_reason="stop",
            tool_calls=None,
        )

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client), \
             patch("agent.transports.codex.ResponsesApiTransport", return_value=mock_transport):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "***",
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
                raw = await resp.text()

        assert resp.status == 200
        payloads = _sse_data_payloads(raw)
        assert payloads[-1] == "[DONE]"
        chunks = [json.loads(item) for item in payloads if item != "[DONE]"]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[1]["choices"][0]["delta"] == {"content": "codex "}
        assert chunks[2]["choices"][0]["delta"] == {"content": "stream"}
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["model"] == "gpt-5.4"
        adapter._run_agent.assert_not_called()
        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["model"] == "gpt-5.4"
        assert kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_codex_responses_streaming_tool_calls_are_adapted_to_chat_completion_sse(self):
        adapter = _proxy_adapter()
        adapter._run_agent = AsyncMock()
        app = _create_app(adapter)

        function_item = SimpleNamespace(
            type="function_call",
            id="fc_read",
            call_id="call_read",
            name="read_file",
            arguments='{"path":"README.md"}',
            status="completed",
        )
        terminal_response = SimpleNamespace(output=[], usage={"input_tokens": 5, "output_tokens": 1})
        stream_events = iter([
            SimpleNamespace(type="response.output_item.added", item=SimpleNamespace(
                type="function_call",
                id="fc_read",
                call_id="call_read",
                name="read_file",
                arguments="",
            )),
            SimpleNamespace(type="response.function_call_arguments.delta", item_id="fc_read", delta='{"path"'),
            SimpleNamespace(type="response.function_call_arguments.delta", item_id="fc_read", delta=':"README.md"}'),
            SimpleNamespace(type="response.output_item.done", item=function_item),
            SimpleNamespace(type="response.completed", response=terminal_response),
        ])
        mock_client = MagicMock()
        mock_client.responses.create.return_value = stream_events
        mock_transport = MagicMock()
        mock_transport.normalize_response.return_value = SimpleNamespace(
            content="",
            finish_reason="tool_calls",
            tool_calls=[SimpleNamespace(id="call_read", function=SimpleNamespace(name="read_file", arguments='{"path":"README.md"}'))],
        )

        with patch("gateway.api_server_provider_proxy.resolve_runtime_provider") as mock_resolve, \
             patch("gateway.api_server_provider_proxy.OpenAI", return_value=mock_client), \
             patch("agent.transports.codex.ResponsesApiTransport", return_value=mock_transport):
            mock_resolve.return_value = {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "***",
            }
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-5.4",
                        "stream": True,
                        "messages": [{"role": "user", "content": "read README"}],
                        "tools": [_read_file_tool()],
                        "parallel_tool_calls": False,
                    },
                )
                raw = await resp.text()

        assert resp.status == 200
        payloads = _sse_data_payloads(raw)
        assert payloads[-1] == "[DONE]"
        chunks = [json.loads(item) for item in payloads if item != "[DONE]"]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[1]["choices"][0]["delta"] == {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": ""},
                }
            ]
        }
        assert chunks[2]["choices"][0]["delta"] == {
            "tool_calls": [{"index": 0, "function": {"arguments": '{"path"'}}]
        }
        assert chunks[3]["choices"][0]["delta"] == {
            "tool_calls": [{"index": 0, "function": {"arguments": ':"README.md"}'}}]
        }
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert mock_client.responses.create.call_args.kwargs["parallel_tool_calls"] is False
