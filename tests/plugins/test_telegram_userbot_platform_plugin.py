"""Behavior tests for the PatchKit Telegram MTProto userbot platform plugin."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult, build_session_key


class FakeTelegramMessage:
    def __init__(
        self,
        *,
        message_id: int = 111,
        chat_id: int = -100777,
        sender_id: int = 123,
        text: str = "hello",
        outgoing: bool = False,
    ) -> None:
        self.id = message_id
        self.chat_id = chat_id
        self.sender_id = sender_id
        self.raw_text = text
        self.message = text
        self.out = outgoing
        self.date = datetime(2026, 7, 11, tzinfo=timezone.utc)
        self.photo: Any = None
        self.document: Any = None
        self.media: Any = None


class FakeNewMessageEvent:
    def __init__(self, message: FakeTelegramMessage, *, private: bool = False) -> None:
        self.message = message
        self.chat_id = message.chat_id
        self.sender_id = message.sender_id
        self.raw_text = message.raw_text
        self.is_private = private


class FakeTelegramClient:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.sent_files: list[dict] = []
        self.edited: list[dict] = []
        self.actions: list[tuple[object, str]] = []
        self.active_actions: set[tuple[object, str]] = set()
        self.downloaded_to: list[str] = []
        self.iterated_media: list[Any] = []
        self.download_content = b"voice"
        self.disconnected = False

    async def send_message(self, entity, message, **kwargs):
        self.sent.append({"entity": entity, "message": message, **kwargs})
        return SimpleNamespace(id=333)

    async def send_file(self, entity, file, **kwargs):
        self.sent_files.append({"entity": entity, "file": file, **kwargs})
        return SimpleNamespace(id=444)

    async def download_media(self, message, *, file):
        self.downloaded_to.append(file)
        path = Path(file) / "voice.ogg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.download_content)
        return str(path)

    async def iter_download(self, media, **kwargs):
        self.iterated_media.append(media)
        yield self.download_content

    async def edit_message(self, entity, message, text, **kwargs):
        self.edited.append({"entity": entity, "message": message, "text": text, **kwargs})
        return SimpleNamespace(id=int(message))

    @asynccontextmanager
    async def action(self, entity, action):
        key = (entity, action)
        self.actions.append(key)
        self.active_actions.add(key)
        try:
            yield
        finally:
            self.active_actions.discard(key)

    async def disconnect(self):
        self.disconnected = True



def test_register_exposes_telegram_userbot_platform_metadata():
    from plugins.platforms.telegram_userbot import register

    calls = []

    class Ctx:
        def register_platform(self, **kwargs):
            calls.append(kwargs)

    register(Ctx())

    assert len(calls) == 1
    entry = calls[0]
    assert entry["name"] == "telegram_userbot"
    assert entry["label"] == "Telegram Userbot"
    assert entry["required_env"] == [
        "TELEGRAM_USERBOT_API_ID",
        "TELEGRAM_USERBOT_API_HASH",
    ]
    assert entry["allowed_users_env"] == "TELEGRAM_USERBOT_ALLOWED_USERS"
    assert entry["allow_all_env"] == "TELEGRAM_USERBOT_ALLOW_ALL_USERS"
    assert entry["max_message_length"] == 4096
    assert "MTProto" in entry["platform_hint"]



def test_dynamic_platform_value_is_accepted_for_telegram_userbot_plugin():
    assert Platform("telegram_userbot").value == "telegram_userbot"



def test_session_work_dir_is_profile_local(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True)
    )

    assert adapter.work_dir == str(tmp_path / "telegram_userbot")
    assert adapter.session_name == "main"
    assert adapter.session_path == tmp_path / "telegram_userbot" / "main.session"


def test_telegram_userbot_suppresses_bot_style_system_messages(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(PlatformConfig(enabled=True))

    assert adapter.suppress_system_messages is True


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/new", "/reset"])
async def test_suppressed_commands_leave_active_session_unchanged(
    command, tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"allow_all_users": True})
    )
    event = adapter._message_to_event(
        FakeNewMessageEvent(
            FakeTelegramMessage(text=command),
            private=True,
        )
    )
    assert event is not None

    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get(
            "group_sessions_per_user", True
        ),
        thread_sessions_per_user=adapter.config.extra.get(
            "thread_sessions_per_user", False
        ),
    )
    guard = asyncio.Event()
    active_task = asyncio.create_task(asyncio.sleep(60))
    pending_event = adapter._message_to_event(
        FakeNewMessageEvent(
            FakeTelegramMessage(message_id=222, text="pending"),
            private=True,
        )
    )
    assert pending_event is not None
    adapter._active_sessions[session_key] = guard
    adapter._session_tasks[session_key] = active_task
    adapter._pending_messages[session_key] = pending_event
    adapter._message_handler = AsyncMock(return_value=None)
    adapter.cancel_session_processing = AsyncMock()

    try:
        await adapter.handle_message(event)

        assert adapter._active_sessions[session_key] is guard
        assert adapter._session_tasks[session_key] is active_task
        assert adapter._pending_messages[session_key] is pending_event
        assert active_task.cancelled() is False
        adapter._message_handler.assert_not_awaited()
        adapter.cancel_session_processing.assert_not_awaited()
    finally:
        active_task.cancel()
        spawned_tasks = list(adapter._background_tasks)
        for task in spawned_tasks:
            task.cancel()
        await asyncio.gather(active_task, *spawned_tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress_system_messages", [True, False])
async def test_background_handler_errors_follow_system_message_capability(
    suppress_system_messages, tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    class AdapterUnderTest(TelegramUserbotAdapter):
        @property
        def suppress_system_messages(self):
            return suppress_system_messages

    adapter = AdapterUnderTest(
        PlatformConfig(enabled=True, extra={"allow_all_users": True})
    )
    event = adapter._message_to_event(
        FakeNewMessageEvent(
            FakeTelegramMessage(text="trigger background failure"),
            private=True,
        )
    )
    assert event is not None
    adapter._message_handler = AsyncMock(
        side_effect=RuntimeError("sensitive internal detail")
    )
    adapter.send = AsyncMock()

    await adapter.handle_message(event)
    background_tasks = list(adapter._background_tasks)
    assert background_tasks
    await asyncio.gather(*background_tasks)

    if suppress_system_messages:
        adapter.send.assert_not_awaited()
    else:
        adapter.send.assert_awaited_once()
        content = adapter.send.await_args.kwargs["content"]
        assert "RuntimeError" in content
        assert "sensitive internal detail" in content


@pytest.mark.asyncio
async def test_unknown_slash_text_reaches_userbot_message_handler(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"allow_all_users": True})
    )
    event = adapter._message_to_event(
        FakeNewMessageEvent(
            FakeTelegramMessage(text="/not-a-hermes-command"),
            private=True,
        )
    )
    assert event is not None
    adapter._message_handler = AsyncMock(return_value=None)

    await adapter.handle_message(event)
    await asyncio.gather(*list(adapter._background_tasks))

    adapter._message_handler.assert_awaited_once_with(event)


def test_validate_config_requires_env_credentials_plus_bootstrap_or_session(
    tmp_path, monkeypatch
):
    profile_root = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(profile_root))
    for key in (
        "TELEGRAM_USERBOT_API_ID",
        "TELEGRAM_USERBOT_API_HASH",
        "TELEGRAM_USERBOT_PHONE",
        "TELEGRAM_USERBOT_SESSION_STRING",
    ):
        monkeypatch.delenv(key, raising=False)

    from plugins.platforms.telegram_userbot.adapter import (
        TelegramUserbotAdapter,
        validate_telegram_userbot_config,
    )

    secret_extra = PlatformConfig(
        enabled=True,
        extra={
            "api_id": "123",
            "api_hash": "hash",
            "phone": "+799****0000",
            "session_string": "opaque-session",
            "two_factor_password": "must-not-be-in-config",
            "2fa_password": "must-not-be-in-config",
        },
    )
    assert validate_telegram_userbot_config(secret_extra) is False
    with pytest.raises(ValueError, match="profile .env"):
        TelegramUserbotAdapter(secret_extra)

    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    assert validate_telegram_userbot_config(PlatformConfig(enabled=True)) is False

    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+799****0000")
    assert validate_telegram_userbot_config(PlatformConfig(enabled=True)) is True

    monkeypatch.delenv("TELEGRAM_USERBOT_PHONE")
    monkeypatch.setenv("TELEGRAM_USERBOT_SESSION_STRING", "opaque-session")
    assert validate_telegram_userbot_config(PlatformConfig(enabled=True)) is True

    monkeypatch.delenv("TELEGRAM_USERBOT_SESSION_STRING")
    work_dir = profile_root / "sessions"
    work_dir.mkdir(parents=True)
    (work_dir / "main.session").write_text("sqlite-ish")
    assert validate_telegram_userbot_config(
        PlatformConfig(
            enabled=True,
            extra={"work_dir": str(work_dir), "session_name": "main"},
        )
    ) is True



def test_message_to_event_maps_identity_and_text():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "allowed_users": ["123"],
            },
        )
    )
    raw = FakeNewMessageEvent(FakeTelegramMessage())

    event = adapter._message_to_event(raw)

    assert event is not None
    assert event.source.platform == Platform("telegram_userbot")
    assert event.source.chat_id == "-100777"
    assert event.source.user_id == "123"
    assert event.text == "hello"
    assert event.message_id == "111"
    assert event.message_type.value == "text"


def test_allowlisted_sender_reaches_gateway_without_bot_pairing(monkeypatch):
    from gateway.run import GatewayRunner
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    for key in (
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "TELEGRAM_USERBOT_ALLOWED_USERS",
        "TELEGRAM_USERBOT_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={"allowed_users": ["123"]},
        )
    )
    event = adapter._message_to_event(
        FakeNewMessageEvent(FakeTelegramMessage(sender_id=123), private=True)
    )
    assert event is not None

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform("telegram_userbot"): adapter}
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_args: False)

    assert runner._is_user_authorized(event.source) is True



def test_message_to_event_rejects_unallowlisted_sender():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "allowed_users": ["123"],
            },
        )
    )

    assert adapter._message_to_event(
        FakeNewMessageEvent(FakeTelegramMessage(sender_id=999))
    ) is None
    assert adapter._message_to_event(
        FakeNewMessageEvent(FakeTelegramMessage(sender_id=123))
    ) is not None



def test_message_to_event_ignores_own_outgoing_messages():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={"allow_all_users": True},
        )
    )

    assert adapter._message_to_event(
        FakeNewMessageEvent(FakeTelegramMessage(outgoing=True))
    ) is None


@pytest.mark.asyncio
async def test_send_text_and_reply_use_mtproto_client():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True)
    )
    client = FakeTelegramClient()
    adapter._client = client

    result = await adapter.send("-100777", "hello", reply_to="55")

    assert isinstance(result, SendResult)
    assert result.success is True
    assert result.message_id == "333"
    assert client.sent == [
        {"entity": -100777, "message": "hello", "reply_to": 55}
    ]


@pytest.mark.asyncio
async def test_edit_message_uses_mtproto_client():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True)
    )
    client = FakeTelegramClient()
    adapter._client = client

    result = await adapter.edit_message("-100777", "55", "updated")

    assert result.success is True
    assert result.message_id == "55"
    assert client.edited == [
        {"entity": -100777, "message": 55, "text": "updated"}
    ]


@pytest.mark.asyncio
async def test_send_typing_enters_mtproto_action_context():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True)
    )
    client = FakeTelegramClient()
    adapter._client = client

    await adapter.send_typing("-100777")
    await asyncio.sleep(0)

    assert client.actions == [(-100777, "typing")]
    assert client.active_actions == {(-100777, "typing")}

    await adapter.disconnect()
    assert client.active_actions == set()


@pytest.mark.asyncio
async def test_human_pacing_keeps_initial_thinking_pause_silent():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "allow_all_users": True,
                "human_pacing_enabled": True,
                "thinking_delay_min_ms": 25,
                "thinking_delay_max_ms": 25,
            },
        )
    )
    client = FakeTelegramClient()
    adapter._client = client

    async def observe_typing(event):
        task = asyncio.create_task(adapter._keep_typing(event.source.chat_id))
        await asyncio.sleep(0.01)
        assert client.actions == []
        await asyncio.sleep(0.03)
        assert client.actions == [(-100777, "typing")]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    adapter.handle_message = observe_typing
    await adapter._handle_new_message(
        FakeNewMessageEvent(FakeTelegramMessage(sender_id=123), private=True)
    )
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_human_pacing_pads_fast_text_response_by_length():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "allow_all_users": True,
                "human_pacing_enabled": True,
                "thinking_delay_min_ms": 1,
                "thinking_delay_max_ms": 1,
                "typing_chars_per_second": 1000,
                "typing_delay_min_ms": 25,
                "typing_delay_max_ms": 25,
            },
        )
    )
    client = FakeTelegramClient()
    adapter._client = client
    elapsed: list[float] = []
    results: list[SendResult] = []

    async def send_fast_reply(event):
        typing_task = asyncio.create_task(adapter._keep_typing(event.source.chat_id))
        await asyncio.sleep(0.005)
        started = asyncio.get_running_loop().time()
        results.append(await adapter.send(event.source.chat_id, "hello"))
        elapsed.append(asyncio.get_running_loop().time() - started)
        typing_task.cancel()
        await asyncio.gather(typing_task, return_exceptions=True)

    adapter.handle_message = send_fast_reply
    await adapter._handle_new_message(
        FakeNewMessageEvent(FakeTelegramMessage(sender_id=123), private=True)
    )

    assert results[0].success is True
    assert elapsed[0] >= 0.02
    assert client.actions == [(-100777, "typing")]
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_human_pacing_excluded_sender_bypasses_artificial_delays():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "allow_all_users": True,
                "human_pacing_enabled": True,
                "human_pacing_excluded_user_ids": ["380342859", 777],
            },
        )
    )
    observed: list[tuple[str | None, bool]] = []

    async def capture_pacing(event):
        observed.append((event.source.user_id, adapter._human_pacing_applies()))

    adapter.handle_message = capture_pacing
    await adapter._handle_new_message(
        FakeNewMessageEvent(FakeTelegramMessage(sender_id=380342859), private=True)
    )
    await adapter._handle_new_message(
        FakeNewMessageEvent(FakeTelegramMessage(sender_id=123), private=True)
    )

    assert adapter.human_pacing_excluded_user_ids == {"380342859", "777"}
    assert observed == [("380342859", False), ("123", True)]


@pytest.mark.asyncio
async def test_queued_follow_up_uses_pending_event_sender_context(monkeypatch):
    from gateway.platforms.base import BasePlatformAdapter
    from plugins.platforms.telegram_userbot import adapter as userbot_module
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={"allow_all_users": True, "human_pacing_enabled": True},
        )
    )
    pending_event = adapter._message_to_event(
        FakeNewMessageEvent(FakeTelegramMessage(sender_id=456), private=True)
    )
    assert pending_event is not None
    observed: list[str | None] = []

    async def capture_context(_self, event, _session_key):
        turn = userbot_module._current_pacing_turn.get()
        observed.append(turn.sender_id if turn is not None else None)
        assert event.source.user_id == "456"

    monkeypatch.setattr(
        BasePlatformAdapter,
        "_process_message_background",
        capture_context,
    )
    outer_token = userbot_module._current_pacing_turn.set(
        userbot_module._PacingTurn(chat_id="-100777", sender_id="123")
    )
    try:
        await adapter._process_message_background(pending_event, "shared-group-session")
        restored_turn = userbot_module._current_pacing_turn.get()
        assert restored_turn is not None
        assert restored_turn.sender_id == "123"
    finally:
        userbot_module._current_pacing_turn.reset(outer_token)

    assert observed == ["456"]


@pytest.mark.asyncio
async def test_human_pacing_does_not_delay_one_off_send_during_active_turn():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "allow_all_users": True,
                "human_pacing_enabled": True,
                "thinking_delay_min_ms": 80,
                "thinking_delay_max_ms": 80,
                "typing_delay_min_ms": 1,
                "typing_delay_max_ms": 1,
            },
        )
    )
    client = FakeTelegramClient()
    adapter._client = client
    turn_started = asyncio.Event()
    release_turn = asyncio.Event()

    async def hold_inbound_turn(event):
        typing_task = asyncio.create_task(adapter._keep_typing(event.source.chat_id))
        await asyncio.sleep(0)
        turn_started.set()
        await release_turn.wait()
        typing_task.cancel()
        await asyncio.gather(typing_task, return_exceptions=True)

    adapter.handle_message = hold_inbound_turn
    inbound_task = asyncio.create_task(
        adapter._handle_new_message(
            FakeNewMessageEvent(FakeTelegramMessage(sender_id=123), private=True)
        )
    )
    await turn_started.wait()

    started = asyncio.get_running_loop().time()
    result = await adapter.send("-100777", "service message")
    elapsed = asyncio.get_running_loop().time() - started

    release_turn.set()
    await inbound_task
    assert result.success is True
    assert elapsed < 0.04
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_human_pacing_isolated_between_concurrent_turns_in_same_chat():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "allow_all_users": True,
                "human_pacing_enabled": True,
                "thinking_delay_min_ms": 60,
                "thinking_delay_max_ms": 60,
                "typing_delay_min_ms": 1,
                "typing_delay_max_ms": 1,
            },
        )
    )
    client = FakeTelegramClient()
    adapter._client = client
    first_ready = asyncio.Event()
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    first_elapsed: list[float] = []

    async def concurrent_reply(event):
        sender_id = event.source.user_id
        typing_task = asyncio.create_task(adapter._keep_typing(event.source.chat_id))
        if sender_id == "123":
            await asyncio.sleep(0.075)
            first_ready.set()
            await second_started.wait()
            started = asyncio.get_running_loop().time()
            await adapter.send(event.source.chat_id, "first reply")
            first_elapsed.append(asyncio.get_running_loop().time() - started)
        else:
            second_started.set()
            await release_second.wait()
        typing_task.cancel()
        await asyncio.gather(typing_task, return_exceptions=True)

    adapter.handle_message = concurrent_reply
    first_task = asyncio.create_task(
        adapter._handle_new_message(
            FakeNewMessageEvent(FakeTelegramMessage(sender_id=123), private=True)
        )
    )
    await first_ready.wait()
    second_task = asyncio.create_task(
        adapter._handle_new_message(
            FakeNewMessageEvent(FakeTelegramMessage(sender_id=456), private=True)
        )
    )
    await second_started.wait()
    await first_task
    release_second.set()
    await second_task

    assert first_elapsed and first_elapsed[0] < 0.04
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_stop_typing_cancels_active_mtproto_action():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(PlatformConfig(enabled=True))
    client = FakeTelegramClient()
    adapter._client = client

    await adapter.send_typing("-100777")
    assert client.active_actions == {(-100777, "typing")}

    await adapter.stop_typing("-100777")

    assert client.active_actions == set()
    assert adapter._typing_tasks == {}
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_paced_turn_defers_stop_typing_until_final_send():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "allow_all_users": True,
                "human_pacing_enabled": True,
                "thinking_delay_min_ms": 1,
                "thinking_delay_max_ms": 1,
                "typing_delay_min_ms": 1,
                "typing_delay_max_ms": 1,
            },
        )
    )
    client = FakeTelegramClient()
    adapter._client = client

    async def finish_generation_then_send(event):
        typing_task = asyncio.create_task(adapter._keep_typing(event.source.chat_id))
        await asyncio.sleep(0.005)
        assert client.active_actions == {(-100777, "typing")}

        await adapter.stop_typing(event.source.chat_id)
        assert client.active_actions == {(-100777, "typing")}

        result = await adapter.send(event.source.chat_id, "final")
        assert result.success is True
        assert client.active_actions == set()
        assert adapter._typing_tasks == {}

        typing_task.cancel()
        await asyncio.gather(typing_task, return_exceptions=True)

    adapter.handle_message = finish_generation_then_send
    await adapter._handle_new_message(
        FakeNewMessageEvent(FakeTelegramMessage(sender_id=123), private=True)
    )
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_paced_turn_without_final_send_cleans_up_typing_action():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "allow_all_users": True,
                "human_pacing_enabled": True,
                "thinking_delay_min_ms": 1,
                "thinking_delay_max_ms": 1,
            },
        )
    )
    client = FakeTelegramClient()
    adapter._client = client

    async def cancel_without_send(event):
        typing_task = asyncio.create_task(adapter._keep_typing(event.source.chat_id))
        await asyncio.sleep(0.005)
        assert client.active_actions == {(-100777, "typing")}
        typing_task.cancel()
        await asyncio.gather(typing_task, return_exceptions=True)
        assert client.active_actions == set()
        assert adapter._typing_tasks == {}

    adapter.handle_message = cancel_without_send
    await adapter._handle_new_message(
        FakeNewMessageEvent(FakeTelegramMessage(sender_id=123), private=True)
    )
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_older_turn_cleanup_does_not_stop_newer_turn_typing():
    from plugins.platforms.telegram_userbot import adapter as userbot_module
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(PlatformConfig(enabled=True))
    client = FakeTelegramClient()
    adapter._client = client
    first_turn = userbot_module._PacingTurn(chat_id="-100777", sender_id="123")
    second_turn = userbot_module._PacingTurn(chat_id="-100777", sender_id="456")

    first_token = userbot_module._current_pacing_turn.set(first_turn)
    try:
        await adapter.send_typing("-100777")
        second_token = userbot_module._current_pacing_turn.set(second_turn)
        try:
            await adapter.send_typing("-100777")
        finally:
            userbot_module._current_pacing_turn.reset(second_token)

        await adapter.stop_typing("-100777")
        assert client.active_actions == {(-100777, "typing")}
    finally:
        userbot_module._current_pacing_turn.reset(first_token)

    await adapter.stop_typing("-100777")
    assert client.active_actions == set()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_cancelling_send_during_pacing_stops_owned_typing_action():
    from plugins.platforms.telegram_userbot import adapter as userbot_module
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "human_pacing_enabled": True,
                "typing_delay_min_ms": 10000,
                "typing_delay_max_ms": 10000,
                "typing_jitter_ratio": 0,
            },
        )
    )
    client = FakeTelegramClient()
    adapter._client = client
    turn = userbot_module._PacingTurn(chat_id="-100777", sender_id="123")
    turn.thinking_until = asyncio.get_running_loop().time()
    turn.typing_started_at = asyncio.get_running_loop().time()

    token = userbot_module._current_pacing_turn.set(turn)
    try:
        await adapter.send_typing("-100777")
        send_task = asyncio.create_task(adapter.send("-100777", "delayed response"))
        await asyncio.sleep(0)
        send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send_task
        assert client.active_actions == set()
        assert adapter._typing_tasks == {}
    finally:
        userbot_module._current_pacing_turn.reset(token)

    await adapter.disconnect()


def test_human_pacing_uses_configured_typing_jitter_ratio():
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={"typing_jitter_ratio": 0.05},
        )
    )

    assert adapter.typing_jitter_ratio == pytest.approx(0.05)


@pytest.mark.parametrize("value", [-0.1, 1.1, "invalid"])
def test_human_pacing_rejects_invalid_typing_jitter_ratio(value):
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"typing_jitter_ratio": value})
    )

    assert adapter.typing_jitter_ratio == pytest.approx(0.15)


def test_session_lock_prevents_two_clients_for_same_file_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "work_dir": str(tmp_path),
            "session_name": "main",
        },
    )
    first = TelegramUserbotAdapter(cfg)
    second = TelegramUserbotAdapter(cfg)

    first._acquire_session_lock()
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            second._acquire_session_lock()
    finally:
        first._release_session_lock()

    second._acquire_session_lock()
    second._release_session_lock()


@pytest.mark.asyncio
async def test_send_document_uses_mtproto_file_delivery(tmp_path):
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    document = tmp_path / "report.txt"
    document.write_text("report")
    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True)
    )
    client = FakeTelegramClient()
    adapter._client = client

    result = await adapter.send_document(
        "-100777",
        str(document),
        caption="Report",
        reply_to="55",
    )

    assert result.success is True
    assert result.message_id == "444"
    assert client.sent_files == [
        {
            "entity": -100777,
            "file": str(document),
            "caption": "Report",
            "reply_to": 55,
        }
    ]


@pytest.mark.asyncio
async def test_inbound_media_is_downloaded_profile_locally_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "allowed_users": ["123"],
            "work_dir": str(tmp_path),
            "download_media": True,
        },
    )
    adapter = TelegramUserbotAdapter(cfg)
    client = FakeTelegramClient()
    adapter._client = client
    message = FakeTelegramMessage(text="")
    message.media = object()
    message.document = SimpleNamespace(mime_type="audio/ogg", size=5)
    raw = FakeNewMessageEvent(message)
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture

    await adapter._handle_new_message(raw)

    assert len(dispatched) == 1
    event = dispatched[0]
    assert event.message_type.value == "audio"
    assert event.media_types == ["audio/ogg"]
    assert len(event.media_urls) == 1
    downloaded = Path(event.media_urls[0])
    assert downloaded.parent == tmp_path / "media"
    assert downloaded.suffix in {".oga", ".ogg"}
    assert downloaded.read_bytes() == b"voice"
    assert client.iterated_media == [message.media]
    assert client.downloaded_to == []


def install_fake_telethon(monkeypatch) -> None:
    events_module = ModuleType("telethon.events")
    setattr(events_module, "NewMessage", lambda **kwargs: object())
    telethon_module = ModuleType("telethon")
    setattr(telethon_module, "events", events_module)
    monkeypatch.setitem(sys.modules, "telethon", telethon_module)
    monkeypatch.setitem(sys.modules, "telethon.events", events_module)


class FakeLifecycleClient:
    def __init__(self, *, fail_get_me: bool = False) -> None:
        self.fail_get_me = fail_get_me
        self.disconnected = False
        self.handlers = []

    def add_event_handler(self, callback, event):
        self.handlers.append((callback, event))

    async def start(self, **kwargs):
        return self

    async def get_me(self):
        if self.fail_get_me:
            raise RuntimeError("get_me failed")
        return SimpleNamespace(id=999)

    async def run_until_disconnected(self):
        return None

    async def disconnect(self):
        self.disconnected = True


class BlockingLifecycleClient(FakeLifecycleClient):
    def __init__(self) -> None:
        super().__init__()
        self.run_entered = asyncio.Event()
        self.stopped = asyncio.Event()

    async def run_until_disconnected(self):
        self.run_entered.set()
        await self.stopped.wait()

    async def disconnect(self):
        self.disconnected = True
        self.stopped.set()


class SlowDisconnectLifecycleClient(BlockingLifecycleClient):
    def __init__(self) -> None:
        super().__init__()
        self.disconnect_entered = asyncio.Event()
        self.release_disconnect = asyncio.Event()

    async def disconnect(self):
        self.disconnect_entered.set()
        await self.release_disconnect.wait()
        self.disconnected = True
        self.stopped.set()


class FutureDisconnectLifecycleClient(BlockingLifecycleClient):
    def disconnect(self):  # type: ignore[override]
        self.disconnected = True
        self.stopped.set()
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        return future


@pytest.mark.asyncio
async def test_disconnect_accepts_future_returned_by_telethon(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+799****0000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    client = FutureDisconnectLifecycleClient()
    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"work_dir": str(tmp_path)})
    )
    adapter._build_client = lambda: client

    assert await adapter.connect() is True
    await adapter.disconnect()

    assert client.disconnected is True
    assert adapter._client is None
    assert adapter._client_task is None
    assert adapter._session_lock_file is None
    assert adapter._running is False


@pytest.mark.asyncio
async def test_connect_waits_for_unexpected_run_loop_teardown(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+79990000000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    first = SlowDisconnectLifecycleClient()
    second = BlockingLifecycleClient()
    clients = [first, second]
    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"work_dir": str(tmp_path)})
    )
    adapter._build_client = lambda: clients.pop(0)

    assert await adapter.connect() is True
    first.stopped.set()
    await asyncio.wait_for(first.disconnect_entered.wait(), timeout=1)

    reconnect = asyncio.create_task(adapter.connect())
    await asyncio.sleep(0)
    assert reconnect.done() is False

    first.release_disconnect.set()
    assert await asyncio.wait_for(reconnect, timeout=1) is True
    assert adapter._client is second
    assert adapter._running is True
    assert len(second.handlers) == 1

    await adapter.disconnect()


class FailGetMeSlowDisconnectClient(SlowDisconnectLifecycleClient):
    async def get_me(self):
        raise RuntimeError("get_me failed")


@pytest.mark.asyncio
async def test_repeated_cancellation_of_normal_error_cleanup_is_fail_safe(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+79990000000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    client = FailGetMeSlowDisconnectClient()
    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"work_dir": str(tmp_path)})
    )
    adapter._build_client = lambda: client

    connect_task = asyncio.create_task(adapter.connect())
    await asyncio.wait_for(client.disconnect_entered.wait(), timeout=1)
    connect_task.cancel()
    await asyncio.sleep(0)
    connect_task.cancel()
    await asyncio.sleep(0)
    completed_before_release = connect_task.done()

    client.release_disconnect.set()
    result = await asyncio.gather(connect_task, return_exceptions=True)

    assert completed_before_release is False
    assert isinstance(result[0], asyncio.CancelledError)
    assert client.disconnected is True
    assert adapter._client is None
    assert adapter._client_task is None
    assert adapter._session_lock_file is None
    assert adapter._running is False


@pytest.mark.asyncio
async def test_repeated_cancellation_of_run_loop_teardown_keeps_barrier_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+799****0000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    client = SlowDisconnectLifecycleClient()
    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"work_dir": str(tmp_path)})
    )
    adapter._build_client = lambda: client
    assert await adapter.connect() is True
    client_task = adapter._client_task
    assert client_task is not None
    await asyncio.wait_for(client.run_entered.wait(), timeout=1)

    client_task.cancel()
    await asyncio.wait_for(client.disconnect_entered.wait(), timeout=1)
    client_task.cancel()
    await asyncio.sleep(0)
    client_task.cancel()
    await asyncio.sleep(0)
    completed_before_release = client_task.done()
    barrier_open_before_release = adapter._teardown_complete.is_set()
    lock_released_before_release = adapter._session_lock_file is None

    client.release_disconnect.set()
    result = await asyncio.gather(client_task, return_exceptions=True)

    assert completed_before_release is False
    assert barrier_open_before_release is False
    assert lock_released_before_release is False
    assert isinstance(result[0], asyncio.CancelledError)
    assert client.disconnected is True
    assert adapter._client is None
    assert adapter._client_task is None
    assert adapter._session_lock_file is None
    assert adapter._running is False
    assert adapter._teardown_complete.is_set() is True


@pytest.mark.asyncio
async def test_repeated_run_loop_cancellation_during_typing_cleanup_is_fail_safe(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+799****0000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import (
        TelegramUserbotAdapter,
        _TypingAction,
    )

    client = SlowDisconnectLifecycleClient()
    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"work_dir": str(tmp_path)})
    )
    adapter._build_client = lambda: client
    assert await adapter.connect() is True
    client_task = adapter._client_task
    assert client_task is not None
    await asyncio.wait_for(client.run_entered.wait(), timeout=1)

    typing_cancelled = asyncio.Event()
    release_typing = asyncio.Event()

    async def slow_typing_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            typing_cancelled.set()
            await release_typing.wait()

    typing_task = asyncio.create_task(slow_typing_cleanup())
    adapter._typing_tasks["chat"] = _TypingAction(owner=None, task=typing_task)

    client_task.cancel()
    await asyncio.wait_for(typing_cancelled.wait(), timeout=1)
    client_task.cancel()
    await asyncio.sleep(0)
    client_task.cancel()
    await asyncio.sleep(0)

    assert client_task.done() is False
    assert adapter._teardown_complete.is_set() is False
    assert adapter._session_lock_file is not None
    assert client.disconnect_entered.is_set() is False

    release_typing.set()
    await asyncio.wait_for(client.disconnect_entered.wait(), timeout=1)
    client.release_disconnect.set()
    result = await asyncio.gather(client_task, return_exceptions=True)

    assert isinstance(result[0], asyncio.CancelledError)
    assert client.disconnected is True
    assert adapter._client is None
    assert adapter._client_task is None
    assert adapter._session_lock_file is None
    assert adapter._running is False
    assert adapter._teardown_complete.is_set() is True


@pytest.mark.asyncio
async def test_disconnect_cancellation_during_typing_cleanup_is_fail_safe(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+79990000000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import (
        TelegramUserbotAdapter,
        _TypingAction,
    )

    first = BlockingLifecycleClient()
    second = BlockingLifecycleClient()
    clients = [first, second]
    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"work_dir": str(tmp_path)})
    )
    adapter._build_client = lambda: clients.pop(0)
    assert await adapter.connect() is True

    typing_cancelled = asyncio.Event()
    release_typing = asyncio.Event()

    async def slow_typing_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            typing_cancelled.set()
            await release_typing.wait()

    typing_task = asyncio.create_task(slow_typing_cleanup())
    adapter._typing_tasks["chat"] = _TypingAction(owner=None, task=typing_task)

    disconnect_task = asyncio.create_task(adapter.disconnect())
    await asyncio.wait_for(typing_cancelled.wait(), timeout=1)
    disconnect_task.cancel()
    await asyncio.sleep(0)
    disconnect_task.cancel()
    await asyncio.sleep(0)
    completed_before_release = disconnect_task.done()

    release_typing.set()
    result = await asyncio.gather(disconnect_task, return_exceptions=True)
    observed = (
        first.disconnected,
        adapter._client,
        adapter._client_task,
        adapter._session_lock_file,
        adapter._running,
    )
    if observed[1] is not None or observed[3] is not None:
        await adapter.disconnect()

    assert completed_before_release is False
    assert isinstance(result[0], asyncio.CancelledError)
    assert observed == (True, None, None, None, False)

    assert await adapter.connect() is True
    assert adapter._client is second
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_disconnect_cancellation_finishes_cleanup_before_unlock(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+79990000000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"work_dir": str(tmp_path)})
    )
    client = SlowDisconnectLifecycleClient()
    adapter._build_client = lambda: client
    assert await adapter.connect() is True

    disconnect_task = asyncio.create_task(adapter.disconnect())
    await asyncio.wait_for(client.disconnect_entered.wait(), timeout=1)
    disconnect_task.cancel()
    client.release_disconnect.set()
    with pytest.raises(asyncio.CancelledError):
        await disconnect_task

    observed = (
        client.disconnected,
        adapter._client,
        adapter._client_task,
        adapter._session_lock_file,
        adapter._running,
    )
    client.stopped.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert observed == (True, None, None, None, False)


class GateStartLifecycleClient(BlockingLifecycleClient):
    def __init__(self) -> None:
        super().__init__()
        self.start_calls = 0
        self.start_entered = asyncio.Event()
        self.release_start = asyncio.Event()

    async def start(self, **kwargs):
        self.start_calls += 1
        self.start_entered.set()
        await self.release_start.wait()
        return self


@pytest.mark.asyncio
async def test_connect_cancellation_cleans_client_and_session_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+79990000000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"work_dir": str(tmp_path)})
    )
    client = GateStartLifecycleClient()
    adapter._build_client = lambda: client

    connect_task = asyncio.create_task(adapter.connect())
    await asyncio.wait_for(client.start_entered.wait(), timeout=1)
    connect_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await connect_task

    assert client.disconnected is True
    assert adapter._client is None
    assert adapter._session_lock_file is None
    assert adapter._running is False


@pytest.mark.asyncio
async def test_repeated_connect_cancellation_finishes_cleanup_before_reconnect(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+79990000000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import (
        TelegramUserbotAdapter,
        _TypingAction,
    )

    first = GateStartLifecycleClient()
    second = BlockingLifecycleClient()
    clients = [first, second]
    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"work_dir": str(tmp_path)})
    )
    adapter._build_client = lambda: clients.pop(0)

    typing_cancelled = asyncio.Event()
    release_typing = asyncio.Event()

    async def slow_typing_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            typing_cancelled.set()
            await release_typing.wait()

    typing_task = asyncio.create_task(slow_typing_cleanup())
    adapter._typing_tasks["chat"] = _TypingAction(owner=None, task=typing_task)

    connect_task = asyncio.create_task(adapter.connect())
    await asyncio.wait_for(first.start_entered.wait(), timeout=1)
    connect_task.cancel()
    await asyncio.wait_for(typing_cancelled.wait(), timeout=1)
    connect_task.cancel()
    await asyncio.sleep(0)

    reconnect_task = asyncio.create_task(adapter.connect())
    await asyncio.sleep(0)
    completed_before_release = connect_task.done()
    reconnect_started_before_release = reconnect_task.done()

    release_typing.set()
    first_result = await asyncio.gather(connect_task, return_exceptions=True)
    assert await asyncio.wait_for(reconnect_task, timeout=1) is True
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert completed_before_release is False
    assert reconnect_started_before_release is False
    assert isinstance(first_result[0], asyncio.CancelledError)
    assert first.disconnected is True
    assert adapter._client is second
    assert adapter._running is True
    assert adapter._session_lock_file is not None
    assert second.disconnected is False

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_parallel_connect_calls_are_serialized(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+79990000000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"work_dir": str(tmp_path)})
    )
    client = GateStartLifecycleClient()
    adapter._build_client = lambda: client

    first = asyncio.create_task(adapter.connect())
    await asyncio.wait_for(client.start_entered.wait(), timeout=1)
    second = asyncio.create_task(adapter.connect())
    await asyncio.sleep(0)
    observed_start_calls = client.start_calls
    client.release_start.set()

    results = await asyncio.gather(first, second)
    observed_handler_count = len(client.handlers)
    await adapter.disconnect()

    assert results == [True, True]
    assert observed_start_calls == 1
    assert client.start_calls == 1
    assert observed_handler_count == 1


@pytest.mark.asyncio
async def test_successful_connect_disconnect_and_reconnect_use_fresh_clients(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+79990000000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    clients = [BlockingLifecycleClient(), BlockingLifecycleClient()]
    adapter = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"work_dir": str(tmp_path)})
    )
    adapter._build_client = lambda: clients.pop(0)

    assert await adapter.connect() is True
    first = adapter._client
    assert first is not None
    assert len(first.handlers) == 1
    assert adapter._running is True
    assert adapter._session_lock_file is not None
    assert (adapter.session_path.parent.stat().st_mode & 0o777) == 0o700
    lock_path = adapter.session_path.with_suffix(".session.lock")
    assert (lock_path.stat().st_mode & 0o777) == 0o600

    await adapter.disconnect()
    assert first.disconnected is True
    assert adapter._running is False
    assert adapter._session_lock_file is None

    assert await adapter.connect() is True
    second = adapter._client
    assert second is not None and second is not first
    assert len(second.handlers) == 1
    await adapter.disconnect()
    assert second.disconnected is True


@pytest.mark.asyncio
async def test_partial_connect_failure_disconnects_client_and_releases_lock(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+79990000000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter
    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "work_dir": str(tmp_path),
            },
        )
    )
    client = FakeLifecycleClient(fail_get_me=True)
    monkeypatch.setattr(adapter, "_build_client", lambda: client)

    assert await adapter.connect() is False
    assert client.disconnected is True
    assert adapter._client is None
    assert adapter._session_lock_file is None
    assert adapter._running is False


@pytest.mark.asyncio
async def test_unexpected_client_loop_exit_clears_connected_state_and_lock(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_USERBOT_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_USERBOT_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_USERBOT_PHONE", "+79990000000")
    install_fake_telethon(monkeypatch)
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter
    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "work_dir": str(tmp_path),
            },
        )
    )
    client = FakeLifecycleClient()
    monkeypatch.setattr(adapter, "_build_client", lambda: client)

    assert await adapter.connect() is True
    for _ in range(20):
        if adapter._client_task is None:
            break
        await asyncio.sleep(0)

    assert adapter._running is False
    assert adapter._client is None
    assert adapter._client_task is None
    assert adapter._session_lock_file is None
    assert client.disconnected is True


def test_session_lock_rejects_profile_escape_symlinks(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    work_dir = tmp_path / "telegram_userbot"
    work_dir.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("unchanged")
    adapter = TelegramUserbotAdapter(PlatformConfig(enabled=True))

    adapter.session_path.symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlink"):
        adapter._acquire_session_lock()
    adapter.session_path.unlink()

    lock_path = adapter.session_path.with_suffix(".session.lock")
    lock_path.symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlink"):
        adapter._acquire_session_lock()
    assert outside.read_text() == "unchanged"


@pytest.mark.asyncio
async def test_media_cache_symlink_outside_profile_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={"allowed_users": ["123"], "download_media": True},
        )
    )
    work_dir = Path(adapter.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-media"
    outside.mkdir()
    (work_dir / "media").symlink_to(outside, target_is_directory=True)
    client = FakeTelegramClient()
    adapter._client = client
    message = FakeTelegramMessage(text="")
    message.media = object()
    message.document = SimpleNamespace(mime_type="audio/ogg", size=5)
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    await adapter._handle_new_message(FakeNewMessageEvent(message))

    assert len(dispatched) == 1
    assert dispatched[0].media_urls == []
    assert client.downloaded_to == []
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("session_name", ["../escape", "/tmp/escape", "a/b"])
def test_session_name_cannot_escape_profile_root(tmp_path, monkeypatch, session_name):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import (
        TelegramUserbotAdapter,
        validate_telegram_userbot_config,
    )

    config = PlatformConfig(enabled=True, extra={"session_name": session_name})
    assert validate_telegram_userbot_config(config) is False
    with pytest.raises(ValueError, match="session_name"):
        TelegramUserbotAdapter(config)


def test_work_dir_cannot_escape_profile_root(tmp_path, monkeypatch):
    profile_root = tmp_path / "profile"
    outside = tmp_path / "outside"
    monkeypatch.setenv("HERMES_HOME", str(profile_root))
    from plugins.platforms.telegram_userbot.adapter import (
        TelegramUserbotAdapter,
        validate_telegram_userbot_config,
    )

    config = PlatformConfig(enabled=True, extra={"work_dir": str(outside)})
    assert validate_telegram_userbot_config(config) is False
    with pytest.raises(ValueError, match="work_dir"):
        TelegramUserbotAdapter(config)


def test_adapter_declares_that_it_splits_long_messages(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    assert TelegramUserbotAdapter.splits_long_messages is True


def test_empty_allowlists_deny_and_allowed_chat_accepts_any_participant(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    denied = TelegramUserbotAdapter(PlatformConfig(enabled=True))
    raw = FakeNewMessageEvent(FakeTelegramMessage(sender_id=999))
    assert denied._message_to_event(raw) is None

    allowed_chat = TelegramUserbotAdapter(
        PlatformConfig(enabled=True, extra={"allowed_chats": ["-100777"]})
    )
    assert allowed_chat._message_to_event(raw) is not None


@pytest.mark.asyncio
async def test_concurrent_media_downloads_cannot_oversubscribe_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    class ConcurrentDownloadClient(FakeTelegramClient):
        def __init__(self):
            super().__init__()
            self.started = 0
            self.gate = asyncio.Event()

        async def download_media(self, message, *, file):
            self.started += 1
            if self.started == 2:
                self.gate.set()
            await self.gate.wait()
            path = Path(file) / f"voice-{self.started}.ogg"
            path.write_bytes(b"12345678")
            return str(path)

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "allowed_users": ["123"],
                "download_media": True,
                "max_media_bytes": 10,
                "max_media_cache_bytes": 10,
                "max_media_cache_files": 1,
            },
        )
    )
    client = ConcurrentDownloadClient()
    adapter._client = client
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    messages = []
    for message_id in (41, 42):
        message = FakeTelegramMessage(text="")
        message.id = message_id
        message.media = object()
        message.document = SimpleNamespace(mime_type="audio/ogg", size=5)
        messages.append(message)

    await asyncio.gather(
        *(adapter._handle_new_message(FakeNewMessageEvent(msg)) for msg in messages)
    )

    media_dir = Path(adapter.work_dir) / "media"
    files = [path for path in media_dir.iterdir() if path.is_file()]
    assert len(files) <= 1
    assert sum(path.stat().st_size for path in files) <= 10
    assert sum(len(event.media_urls) for event in dispatched) <= 1


@pytest.mark.asyncio
async def test_actual_media_bytes_cannot_exceed_underreported_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "allowed_users": ["123"],
                "download_media": True,
                "max_media_bytes": 10,
                "max_media_cache_bytes": 100,
            },
        )
    )
    client = FakeTelegramClient()
    client.download_content = b"12345678901"
    adapter._client = client
    message = FakeTelegramMessage(text="")
    message.media = object()
    message.document = SimpleNamespace(mime_type="audio/ogg", size=5)
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    await adapter._handle_new_message(FakeNewMessageEvent(message))

    assert len(dispatched) == 1
    assert dispatched[0].media_urls == []
    media_dir = Path(adapter.work_dir) / "media"
    assert list(media_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_oversized_inbound_media_is_rejected_before_download(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram_userbot.adapter import TelegramUserbotAdapter

    adapter = TelegramUserbotAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "allowed_users": ["123"],
                "download_media": True,
                "max_media_bytes": 10,
                "max_media_cache_bytes": 100,
            },
        )
    )
    client = FakeTelegramClient()
    adapter._client = client
    message = FakeTelegramMessage(text="")
    message.media = object()
    message.document = SimpleNamespace(mime_type="audio/ogg", size=11)
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    await adapter._handle_new_message(FakeNewMessageEvent(message))

    assert len(dispatched) == 1
    assert dispatched[0].media_urls == []
    assert client.downloaded_to == []
