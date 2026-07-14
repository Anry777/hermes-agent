"""Tests for the bundled MAX gateway platform plugin."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult


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
    def __init__(self):
        self.posts = []
        self.puts = []

    async def put(self, url, *, params=None, json=None, headers=None, **kwargs):
        self.puts.append({
            "url": url,
            "params": params or {},
            "json": json or {},
            "headers": headers or {},
            "kwargs": kwargs,
        })
        return FakeResponse({"message": {"id": str((params or {}).get("message_id") or f"msg-{len(self.puts)}")}})

    async def post(self, url, *, params=None, json=None, headers=None, **kwargs):
        self.posts.append({
            "url": url,
            "params": params or {},
            "json": json or {},
            "headers": headers or {},
            "kwargs": kwargs,
        })
        return FakeResponse({"message": {"id": f"msg-{len(self.posts)}"}})


def _callback_update(
    payload="ea:once:1",
    *,
    callback_id="cb-1",
    sender_id=123,
    chat_id=777,
    update_id="upd-cb-1",
):
    return {
        "update_id": update_id,
        "update_type": "message_callback",
        "timestamp": 1710000000,
        "callback": {
            "callback_id": callback_id,
            "payload": payload,
            "user": {"user_id": sender_id, "first_name": "Alice", "username": "alice", "is_bot": False},
            "message": {
                "recipient": {"chat_id": chat_id, "type": "chat", "title": "Ops"},
                "body": {"text": "approval"},
            },
        },
    }


def test_register_exposes_max_platform_metadata():
    from plugins.platforms.max import register

    calls = []

    class Ctx:
        def register_platform(self, **kwargs):
            calls.append(kwargs)

    register(Ctx())

    assert len(calls) == 1
    entry = calls[0]
    assert entry["name"] == "max"
    assert entry["label"] == "MAX"
    assert entry["required_env"] == ["MAX_BOT_TOKEN"]
    assert entry["allowed_users_env"] == "MAX_ALLOWED_USERS"
    assert entry["allow_all_env"] == "MAX_ALLOW_ALL_USERS"
    assert entry["max_message_length"] == 4000
    assert "MAX messenger" in entry["platform_hint"]
    assert "MEDIA:/path" in entry["platform_hint"]


def test_dynamic_platform_value_is_accepted_for_bundled_max_plugin():
    assert Platform("max").value == "max"


def test_validate_config_accepts_config_token_or_env(monkeypatch):
    from plugins.platforms.max.adapter import validate_max_config

    monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)
    assert validate_max_config(PlatformConfig(enabled=True)) is False
    assert validate_max_config(PlatformConfig(enabled=True, token="from-config")) is True
    assert validate_max_config(PlatformConfig(enabled=True, extra={"token": "from-extra"})) is True

    monkeypatch.setenv("MAX_BOT_TOKEN", "from-env")
    assert validate_max_config(PlatformConfig(enabled=True)) is True


@pytest.mark.asyncio
async def test_send_text_uses_max_markdown_format_and_raw_authorization():
    from plugins.platforms.max.adapter import MaxAdapter

    adapter = MaxAdapter(PlatformConfig(enabled=True, token="max-token", extra={"transport": "polling"}))
    client = RecordingClient()
    adapter._client = client

    result = await adapter.send("chat:777", "**hello**", metadata={"notify": False})

    assert isinstance(result, SendResult)
    assert result.success is True
    assert client.posts[0]["url"].endswith("/messages")
    assert client.posts[0]["params"] == {"chat_id": "777"}
    assert client.posts[0]["headers"]["Authorization"] == "max-token"
    assert "Bearer" not in client.posts[0]["headers"]["Authorization"]
    assert client.posts[0]["json"] == {
        "text": "**hello**",
        "notify": False,
        "format": "markdown",
    }


@pytest.mark.asyncio
async def test_send_typing_uses_max_group_action_endpoint():
    from plugins.platforms.max.adapter import MaxAdapter

    adapter = MaxAdapter(PlatformConfig(enabled=True, token="max-token", extra={"transport": "polling"}))
    client = RecordingClient()
    adapter._client = client

    await adapter.send_typing("chat:777")

    assert len(client.posts) == 1
    assert client.posts[0]["url"].endswith("/chats/777/actions")
    assert client.posts[0]["params"] == {}
    assert client.posts[0]["headers"]["Authorization"] == "max-token"
    assert "Bearer" not in client.posts[0]["headers"]["Authorization"]
    assert client.posts[0]["json"] == {"action": "typing_on"}


@pytest.mark.asyncio
async def test_send_typing_ignores_user_targets():
    from plugins.platforms.max.adapter import MaxAdapter

    adapter = MaxAdapter(PlatformConfig(enabled=True, token="max-token", extra={"transport": "polling"}))
    client = RecordingClient()
    adapter._client = client

    await adapter.send_typing("user:123")
    await adapter.send_typing("123", metadata={"target_type": "user"})

    assert client.posts == []


@pytest.mark.asyncio
async def test_edit_message_uses_official_put_messages_endpoint():
    from plugins.platforms.max.adapter import MaxAdapter

    adapter = MaxAdapter(PlatformConfig(enabled=True, token="max-token", extra={"transport": "polling"}))
    client = RecordingClient()
    adapter._client = client

    result = await adapter.edit_message("777", "msg-123", "**updated progress**", finalize=True)

    assert result.success is True
    assert result.message_id == "msg-123"
    assert client.puts == [
        {
            "url": "https://platform-api.max.ru/messages",
            "params": {"message_id": "msg-123"},
            "json": {"text": "**updated progress**", "format": "markdown"},
            "headers": {"Authorization": "max-token", "Content-Type": "application/json"},
            "kwargs": {},
        }
    ]


@pytest.mark.asyncio
async def test_edit_message_preserves_existing_attachments_by_default():
    from plugins.platforms.max.adapter import MaxAdapter

    adapter = MaxAdapter(PlatformConfig(enabled=True, token="max-token", extra={"transport": "polling"}))
    client = RecordingClient()
    adapter._client = client

    result = await adapter.edit_message("777", "msg-123", "updated")

    assert result.success is True
    assert "attachments" not in client.puts[0]["json"]


def test_max_tool_progress_uses_edit_path_when_available_with_permanent_fallback_flag():
    from gateway.run import _adapter_can_render_tool_progress, _adapter_uses_permanent_tool_progress
    from plugins.platforms.max.adapter import MaxAdapter

    adapter = MaxAdapter(PlatformConfig(enabled=True, token="max-token", extra={"transport": "polling"}))

    assert adapter.SUPPORTS_PROGRESS_MESSAGES_WITHOUT_EDIT is True
    assert _adapter_can_render_tool_progress(adapter) is True
    assert _adapter_uses_permanent_tool_progress(adapter) is False


@pytest.mark.asyncio
async def test_send_exec_approval_uses_max_inline_callback_keyboard_without_command_payload():
    from plugins.platforms.max.adapter import MaxAdapter

    adapter = MaxAdapter(PlatformConfig(enabled=True, token="max-token", extra={"transport": "polling"}))
    client = RecordingClient()
    adapter._client = client

    result = await adapter.send_exec_approval(
        "777",
        command="rm -rf /tmp/secret",
        session_key="max:777",
        description="dangerous test command",
        metadata={"target_type": "chat"},
    )

    assert result.success is True
    assert result.message_id == "msg-1"
    assert adapter._approval_state == {1: "max:777"}
    post = client.posts[0]
    assert post["url"].endswith("/messages")
    assert post["params"] == {"chat_id": "777"}
    assert post["headers"]["Authorization"] == "max-token"
    body = post["json"]
    assert "Command Approval Required" in body["text"]
    assert "dangerous test command" in body["text"]
    assert "rm -rf /tmp/secret" in body["text"]
    keyboard = body["attachments"][0]
    assert keyboard["type"] == "inline_keyboard"
    buttons = keyboard["payload"]["buttons"]
    assert [[button["text"] for button in row] for row in buttons] == [
        ["✅ Allow Once", "✅ Session"],
        ["✅ Always", "❌ Deny"],
    ]
    payloads = [button["payload"] for row in buttons for button in row]
    assert payloads == ["ea:once:1", "ea:session:1", "ea:always:1", "ea:deny:1"]
    assert all("rm -rf" not in payload for payload in payloads)


@pytest.mark.asyncio
async def test_message_callback_resolves_gateway_approval_and_answers_callback():
    from plugins.platforms.max.adapter import MaxAdapter

    adapter = MaxAdapter(
        PlatformConfig(enabled=True, token="max-token", extra={"transport": "polling", "allow_from": "123"})
    )
    client = RecordingClient()
    adapter._client = client
    adapter._approval_state[1] = "max:777"

    with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
        await adapter._handle_update(_callback_update(payload="ea:session:1", callback_id="cb-123", sender_id=123))

    resolve.assert_called_once_with("max:777", "session")
    assert adapter._approval_state == {}
    assert client.posts == [
        {
            "url": "https://platform-api.max.ru/answers",
            "params": {"callback_id": "cb-123"},
            "json": {
                "notification": "✅ Approved for session",
                "message": {
                    "text": "✅ Approved for session by Alice",
                    "attachments": [],
                    "format": "markdown",
                },
            },
            "headers": {"Authorization": "max-token", "Content-Type": "application/json"},
            "kwargs": {},
        }
    ]


@pytest.mark.asyncio
async def test_unauthorized_message_callback_does_not_resolve_gateway_approval():
    from plugins.platforms.max.adapter import MaxAdapter

    adapter = MaxAdapter(
        PlatformConfig(enabled=True, token="max-token", extra={"transport": "polling", "allow_from": "123"})
    )
    client = RecordingClient()
    adapter._client = client
    adapter._approval_state[1] = "max:777"

    with patch("tools.approval.resolve_gateway_approval") as resolve:
        await adapter._handle_update(_callback_update(payload="ea:always:1", callback_id="cb-bad", sender_id=999))

    resolve.assert_not_called()
    assert adapter._approval_state == {1: "max:777"}
    assert client.posts == [
        {
            "url": "https://platform-api.max.ru/answers",
            "params": {"callback_id": "cb-bad"},
            "json": {"notification": "⛔ You are not authorized to approve commands."},
            "headers": {"Authorization": "max-token", "Content-Type": "application/json"},
            "kwargs": {},
        }
    ]


def test_update_to_event_maps_inbound_image_and_file_attachments():
    from plugins.platforms.max.adapter import MaxAdapter

    adapter = MaxAdapter(PlatformConfig(enabled=True, token="max-token", extra={"transport": "polling"}))
    update = {
        "update_id": "upd-1",
        "update_type": "message_created",
        "timestamp": 1710000000,
        "message": {
            "sender": {"user_id": 123, "first_name": "Alice", "username": "alice", "is_bot": False},
            "recipient": {"chat_id": 777, "type": "chat", "title": "Ops"},
            "timestamp": 1710000000,
            "body": {
                "text": "see attachments",
                "attachments": [
                    {"type": "image", "payload": {"url": "https://i.oneme.ru/image.jpg"}},
                    {"type": "file", "payload": {"url": "https://files.example/doc.pdf", "filename": "doc.pdf"}},
                ],
            },
        },
    }

    event = adapter._update_to_event(update)

    assert event is not None
    assert event.source.platform == Platform("max")
    assert event.message_type.value == "photo"
    assert event.media_urls == ["https://i.oneme.ru/image.jpg", "https://files.example/doc.pdf"]
    assert event.media_types == ["image/jpeg", "application/pdf"]
    assert event.raw_message["update_id"] == "upd-1"


def test_update_to_event_maps_official_audio_attachment_to_audio_event():
    from plugins.platforms.max.adapter import MaxAdapter

    adapter = MaxAdapter(PlatformConfig(enabled=True, token="max-token", extra={"transport": "polling"}))
    update = {
        "update_id": "upd-audio",
        "update_type": "message_created",
        "timestamp": 1710000000,
        "message": {
            "sender": {"user_id": 123, "first_name": "Alice", "username": "alice", "is_bot": False},
            "recipient": {"chat_id": 777, "type": "chat", "title": "Ops"},
            "timestamp": 1710000000,
            "body": {
                "text": "",
                "attachments": [
                    {
                        "type": "audio",
                        "transcription": "recognized voice text",
                        "payload": {"url": "https://files.example/voice", "token": "audio-token"},
                    }
                ],
            },
        },
    }

    event = adapter._update_to_event(update)

    assert event is not None
    assert event.text == "recognized voice text"
    assert event.message_type.value == "audio"
    assert event.media_urls == ["https://files.example/voice"]
    assert event.media_types == ["audio/mpeg"]


def test_update_to_event_preserves_official_audio_mime_from_payload():
    from plugins.platforms.max.adapter import MaxAdapter

    adapter = MaxAdapter(PlatformConfig(enabled=True, token="max-token", extra={"transport": "polling"}))
    update = {
        "update_id": "upd-audio-ogg",
        "update_type": "message_created",
        "timestamp": 1710000000,
        "message": {
            "sender": {"user_id": 123, "first_name": "Alice", "username": "alice", "is_bot": False},
            "recipient": {"chat_id": 777, "type": "chat", "title": "Ops"},
            "timestamp": 1710000000,
            "body": {
                "text": "",
                "attachments": [
                    {
                        "type": "audio",
                        "payload": {
                            "url": "https://files.example/voice.ogg",
                            "token": "audio-token",
                            "mime_type": "audio/ogg",
                        },
                    }
                ],
            },
        },
    }

    event = adapter._update_to_event(update)

    assert event is not None
    assert event.message_type.value == "audio"
    assert event.media_urls == ["https://files.example/voice.ogg"]
    assert event.media_types == ["audio/ogg"]
