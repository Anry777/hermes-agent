"""Tests for the MAX messenger gateway adapter."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.base import MessageType
from tools.send_message_tool import _parse_target_ref, _send_to_platform


class FakeResponse:
    def __init__(self, payload=None, *, status_code=200):
        self._payload = payload or {}
        self.status_code = status_code
        self.text = str(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")


class RecordingClient:
    def __init__(self, *, get_responses=None):
        self.posts = []
        self.gets = []
        self.closed = False
        self._get_responses = list(get_responses or [])

    async def post(self, url, *, params=None, json=None, headers=None):
        self.posts.append({"url": url, "params": params or {}, "json": json or {}, "headers": headers or {}})
        if url.endswith("/subscriptions"):
            return FakeResponse({"success": True})
        return FakeResponse({"message": {"id": f"msg-{len(self.posts)}"}})

    async def get(self, url, *, params=None, headers=None):
        self.gets.append({"url": url, "params": params or {}, "headers": headers or {}})
        if self._get_responses:
            return self._get_responses.pop(0)
        return FakeResponse({"marker": None, "updates": []})

    async def aclose(self):
        self.closed = True


class FakeWebhookRequest:
    def __init__(self, payload, *, secret=None):
        self._payload = payload
        self.headers = {}
        if secret is not None:
            self.headers["X-Max-Bot-Api-Secret"] = secret

    async def json(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


def _make_adapter(token="max-token", **extra):
    from gateway.platforms.max import MaxAdapter

    return MaxAdapter(PlatformConfig(enabled=True, token=token, extra=extra))


def _message_update(text="hello MAX", *, update_id="upd-1", sender_id=123, chat_id=777):
    return {
        "update_id": update_id,
        "update_type": "message_created",
        "timestamp": 1710000000,
        "message": {
            "sender": {"user_id": sender_id, "first_name": "Alice", "username": "alice", "is_bot": False},
            "recipient": {"chat_id": chat_id, "type": "chat", "title": "Ops"},
            "timestamp": 1710000000,
            "body": {"text": text},
        },
    }


async def _drain_adapter_tasks(adapter, *, max_rounds=10):
    for _ in range(max_rounds):
        tasks = [task for task in getattr(adapter, "_background_tasks", set()) if not task.done()]
        if not tasks:
            await asyncio.sleep(0)
            tasks = [task for task in getattr(adapter, "_background_tasks", set()) if not task.done()]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)


class TestMaxConfigLoading:
    def test_apply_env_overrides_enables_webhook_first_max_with_header_token_and_home_channel(self, monkeypatch):
        monkeypatch.setenv("MAX_BOT_TOKEN", "max-token")
        monkeypatch.setenv("MAX_ALLOWED_USERS", "123,456")
        monkeypatch.setenv("MAX_ALLOW_ALL_USERS", "false")
        monkeypatch.setenv("MAX_HOME_CHANNEL", "777")
        monkeypatch.setenv("MAX_HOME_CHANNEL_NAME", "MAX Home")
        monkeypatch.setenv("MAX_WEBHOOK_PUBLIC_URL", "https://bot.example.com/max-webhook")
        monkeypatch.setenv("MAX_WEBHOOK_SECRET", "secret_123")
        monkeypatch.setenv("MAX_WEBHOOK_PATH", "max-webhook")
        monkeypatch.setenv("MAX_WEBHOOK_PORT", "8647")
        monkeypatch.setenv("MAX_UPDATE_TYPES", "message_created,bot_started")
        monkeypatch.setenv("MAX_AUTO_SUBSCRIBE", "true")
        monkeypatch.setenv("MAX_TRANSPORT", "polling")
        monkeypatch.setenv("MAX_POLL_TIMEOUT", "7")
        monkeypatch.setenv("MAX_POLL_IDLE_SLEEP", "0.25")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.MAX in config.platforms
        platform_config = config.platforms[Platform.MAX]
        assert platform_config.enabled is True
        assert platform_config.token == "max-token"
        assert platform_config.extra["allow_from"] == "123,456"
        assert platform_config.extra["allow_all_users"] is False
        assert platform_config.extra["webhook_public_url"] == "https://bot.example.com/max-webhook"
        assert platform_config.extra["webhook_secret"] == "secret_123"
        assert platform_config.extra["webhook_path"] == "max-webhook"
        assert platform_config.extra["webhook_port"] == 8647
        assert platform_config.extra["update_types"] == "message_created,bot_started"
        assert platform_config.extra["auto_subscribe"] is True
        assert platform_config.extra["transport"] == "polling"
        assert platform_config.extra["poll_timeout"] == 7
        assert platform_config.extra["poll_idle_sleep"] == 0.25
        assert platform_config.home_channel == HomeChannel(Platform.MAX, "777", "MAX Home")
        assert config.get_connected_platforms() == [Platform.MAX]

    def test_check_requirements_requires_token(self, monkeypatch):
        from gateway.platforms.max import check_max_requirements

        monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)
        assert check_max_requirements() is False
        monkeypatch.setenv("MAX_BOT_TOKEN", "max-token")
        assert check_max_requirements() is True

    def test_polling_transport_does_not_require_aiohttp_webhook_dependency(self, monkeypatch):
        from gateway.platforms import max as max_platform

        monkeypatch.setenv("MAX_BOT_TOKEN", "max-token")
        monkeypatch.setenv("MAX_TRANSPORT", "polling")
        monkeypatch.setattr(max_platform, "AIOHTTP_AVAILABLE", False)
        assert max_platform.check_max_requirements() is True

        monkeypatch.setenv("MAX_TRANSPORT", "webhook")
        assert max_platform.check_max_requirements() is False


class TestMaxAdapter:
    def test_headers_use_raw_authorization_token_not_query_token_or_bearer(self):
        adapter = _make_adapter(token="max-token")

        headers = adapter._headers()

        assert headers["Authorization"] == "max-token"
        assert headers["Content-Type"] == "application/json"
        assert "Bearer" not in headers["Authorization"]

    def test_subscription_payload_matches_official_webhook_api_and_secret_header_model(self):
        adapter = _make_adapter(
            webhook_public_url="https://bot.example.com/max-webhook",
            webhook_secret="secret_123",
            update_types="message_created,bot_started",
        )

        assert adapter._subscription_payload() == {
            "url": "https://bot.example.com/max-webhook",
            "update_types": ["message_created", "bot_started"],
            "secret": "secret_123",
        }

    def test_subscription_rejects_non_https_or_explicit_port_public_urls(self):
        with pytest.raises(ValueError, match="HTTPS"):
            _make_adapter(webhook_public_url="http://bot.example.com/max-webhook")._subscription_payload()
        with pytest.raises(ValueError, match="port 443"):
            _make_adapter(webhook_public_url="https://bot.example.com:8443/max-webhook")._subscription_payload()

    @pytest.mark.asyncio
    async def test_subscribe_posts_to_official_subscriptions_endpoint(self):
        adapter = _make_adapter(
            token="max-token",
            webhook_public_url="https://bot.example.com/max-webhook",
            webhook_secret="secret_123",
        )
        client = RecordingClient()
        adapter._client = client

        result = await adapter._subscribe_webhook()

        assert result == {"success": True}
        assert client.posts == [
            {
                "url": "https://platform-api.max.ru/subscriptions",
                "params": {},
                "json": {
                    "url": "https://bot.example.com/max-webhook",
                    "update_types": ["message_created", "bot_started"],
                    "secret": "secret_123",
                },
                "headers": {"Authorization": "max-token", "Content-Type": "application/json"},
            }
        ]

    @pytest.mark.asyncio
    async def test_poll_once_gets_official_updates_endpoint_and_advances_marker(self):
        adapter = _make_adapter(token="max-token", transport="polling")
        client = RecordingClient(get_responses=[FakeResponse({"marker": 42, "updates": [_message_update(update_id="poll-1")]})])
        adapter._client = client

        updates = await adapter._poll_once(timeout=1)

        assert updates == [_message_update(update_id="poll-1")]
        assert adapter._poll_marker == 42
        assert client.gets == [
            {
                "url": "https://platform-api.max.ru/updates",
                "params": {"timeout": 1},
                "headers": {"Authorization": "max-token", "Content-Type": "application/json"},
            }
        ]

        await adapter._poll_once(timeout=1)
        assert client.gets[-1]["params"] == {"timeout": 1, "marker": 42}

    @pytest.mark.asyncio
    async def test_polling_updates_reuse_webhook_event_conversion_dedup_and_dispatch(self):
        adapter = _make_adapter(token="max-token", transport="polling")
        client = RecordingClient(get_responses=[FakeResponse({"marker": 100, "updates": [
            _message_update(text="from polling", update_id="same"),
            _message_update(text="from polling", update_id="same"),
        ]})])
        adapter._client = client
        calls = []

        async def handler(event):
            calls.append((event.text, event.source.chat_id, event.source.user_id))

        adapter.set_message_handler(handler)

        for update in await adapter._poll_once(timeout=1):
            await adapter._handle_update(update)
        await _drain_adapter_tasks(adapter)

        assert calls == [("from polling", "777", "123")]

    @pytest.mark.asyncio
    async def test_poll_once_uses_configured_default_timeout(self):
        adapter = _make_adapter(token="max-token", transport="polling", poll_timeout=7)
        client = RecordingClient(get_responses=[FakeResponse({"marker": None, "updates": []})])
        adapter._client = client

        updates = await adapter._poll_once()

        assert updates == []
        assert client.gets == [
            {
                "url": "https://platform-api.max.ru/updates",
                "params": {"timeout": 7},
                "headers": {"Authorization": "max-token", "Content-Type": "application/json"},
            }
        ]

    @pytest.mark.asyncio
    async def test_send_posts_text_to_chat_id_query_with_modern_body_fields(self):
        adapter = _make_adapter(token="max-token")
        client = RecordingClient()
        adapter._client = client

        result = await adapter.send("777", "**hello**", metadata={"target_type": "chat", "notify": False, "format": "markdown"})

        assert result.success is True
        assert result.message_id == "msg-1"
        assert client.posts == [
            {
                "url": "https://platform-api.max.ru/messages",
                "params": {"chat_id": "777"},
                "json": {"text": "**hello**", "notify": False, "format": "markdown"},
                "headers": {"Authorization": "max-token", "Content-Type": "application/json"},
            }
        ]

    @pytest.mark.asyncio
    async def test_send_can_target_user_id_explicitly(self):
        adapter = _make_adapter(token="max-token")
        client = RecordingClient()
        adapter._client = client

        result = await adapter.send("123", "hello", metadata={"target_type": "user"})

        assert result.success is True
        assert client.posts[0]["params"] == {"user_id": "123"}

    @pytest.mark.asyncio
    async def test_long_text_is_chunked_at_official_4000_character_limit(self):
        adapter = _make_adapter(token="max-token")
        client = RecordingClient()
        adapter._client = client

        result = await adapter.send("777", "x" * 4100, metadata={"target_type": "chat"})

        assert result.success is True
        assert len(client.posts) == 2
        assert all(len(call["json"]["text"]) <= 4000 for call in client.posts)

    def test_message_created_update_converts_official_message_body_to_event(self):
        adapter = _make_adapter(bot_user_id="999")

        event = adapter._update_to_event(_message_update())

        assert event is not None
        assert event.text == "hello MAX"
        assert event.message_type == MessageType.TEXT
        assert event.source.platform == Platform.MAX
        assert event.source.chat_id == "777"
        assert event.source.chat_type == "group"
        assert event.source.chat_name == "Ops"
        assert event.source.user_id == "123"
        assert event.source.user_name == "Alice"

    @pytest.mark.asyncio
    async def test_webhook_requires_x_max_bot_api_secret_when_configured_and_acks_quickly(self):
        adapter = _make_adapter(webhook_secret="secret_123")
        calls = []

        async def handler(event):
            calls.append(event.text)

        adapter.set_message_handler(handler)
        bad = await adapter._handle_webhook(FakeWebhookRequest(_message_update(), secret="wrong"))
        assert bad.status == 401

        ok = await adapter._handle_webhook(FakeWebhookRequest(_message_update(), secret="secret_123"))
        assert ok.status == 200
        await _drain_adapter_tasks(adapter)
        assert calls == ["hello MAX"]

    @pytest.mark.asyncio
    async def test_webhook_rejects_invalid_json_before_dispatch(self):
        adapter = _make_adapter(webhook_secret="secret_123")
        calls = []
        adapter.set_message_handler(lambda event: calls.append(event.text))

        response = await adapter._handle_webhook(FakeWebhookRequest(ValueError("bad json"), secret="secret_123"))

        assert response.status == 400
        assert calls == []

    @pytest.mark.asyncio
    async def test_self_messages_and_duplicate_webhook_updates_are_not_dispatched(self):
        adapter = _make_adapter(bot_user_id="999")
        calls = []

        async def handler(event):
            calls.append(event.text)

        adapter.set_message_handler(handler)
        await adapter._handle_update(_message_update(text="dedup me", update_id="same"))
        await _drain_adapter_tasks(adapter)
        await adapter._handle_update(_message_update(text="dedup me", update_id="same"))
        await _drain_adapter_tasks(adapter)
        await adapter._handle_update(_message_update(text="from bot", update_id="bot", sender_id=999))
        await _drain_adapter_tasks(adapter)

        assert calls == ["dedup me"]


class TestMaxSendMessageToolIntegration:
    def test_max_target_parse_defaults_to_chat_id_and_accepts_explicit_user_prefix(self):
        assert _parse_target_ref("max", "777") == ("777", None, True)
        assert _parse_target_ref("max", "user:123") == ("user:123", None, True)

    def test_send_to_platform_routes_max_and_chunks_to_4000(self):
        send = AsyncMock(return_value={"success": True, "message_id": "1"})
        with patch("tools.send_message_tool._send_max", send):
            result = asyncio.run(
                _send_to_platform(
                    Platform.MAX,
                    SimpleNamespace(enabled=True, token="max-token", extra={}),
                    "777",
                    "x" * 4100,
                )
            )

        assert result["success"] is True
        assert send.await_count == 2
        assert all(len(call.args[2]) <= 4000 for call in send.await_args_list)
