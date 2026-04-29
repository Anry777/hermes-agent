"""Tests for API Server provider_proxy mode."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware


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
                    "models": [
                        {
                            "id": "openrouter/test-model",
                            "provider": "openrouter",
                            "model": "upstream/test-model",
                        },
                        {
                            "id": "codex/gpt-5.4",
                            "provider": "openai-codex",
                            "model": "gpt-5.4",
                        },
                    ]
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
            "codex/gpt-5.4",
        ]
        assert data["data"][0]["owned_by"] == "openrouter"
        assert data["data"][1]["owned_by"] == "openai-codex"


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
                        "model": "codex/gpt-5.4",
                        "messages": [
                            {"role": "system", "content": "be concise"},
                            {"role": "user", "content": "hello"},
                        ],
                        "max_completion_tokens": 32,
                    },
                )
                data = await resp.json()

        assert resp.status == 200
        assert data["model"] == "codex/gpt-5.4"
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
    async def test_streaming_is_explicitly_rejected_in_initial_proxy_mode(self):
        adapter = _proxy_adapter()
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "openrouter/test-model",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["code"] == "unsupported_streaming"
