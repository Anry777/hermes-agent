"""Tests for the bundled MAX userbot platform plugin."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult


class FakePyMaxMessage:
    def __init__(
        self,
        *,
        id=111,
        chat_id=777,
        sender=123,
        text="hello",
        time=1710000000,
        type="message",
        attaches=None,
    ):
        self.id = id
        self.chat_id = chat_id
        self.sender = sender
        self.text = text
        self.time = time
        self.type = type
        self.attaches = attaches or []
        self.answers = []
        self.replies = []
        self.edits = []
        self.deleted = False
        self.read_called = False

    async def answer(self, text, attachments=None, notify=True, reply_to=None):
        self.answers.append({"text": text, "attachments": attachments, "notify": notify, "reply_to": reply_to})
        return FakePyMaxMessage(id=222, chat_id=self.chat_id, sender=999, text=text)

    async def reply(self, text, attachments=None, notify=True):
        self.replies.append({"text": text, "attachments": attachments, "notify": notify})
        return FakePyMaxMessage(id=223, chat_id=self.chat_id, sender=999, text=text)

    async def edit(self, text, attachments=None):
        self.edits.append({"text": text, "attachments": attachments})
        self.text = text
        return self

    async def read(self):
        self.read_called = True
        return SimpleNamespace(mark=self.id)


class FakePyMaxClient:
    def __init__(self):
        self.sent_messages = []
        self.edits = []
        self.closed = False
        self.started = False
        self.message_handlers = []
        self.raw_handlers = []

    def on_message(self, *filters):
        def deco(fn):
            self.message_handlers.append(fn)
            return fn
        return deco

    def on_raw(self, *filters):
        def deco(fn):
            self.raw_handlers.append(fn)
            return fn
        return deco

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True

    async def send_message(self, chat_id, text, reply_to=None, attachments=None, notify=True):
        self.sent_messages.append(
            {"chat_id": chat_id, "text": text, "reply_to": reply_to, "attachments": attachments, "notify": notify}
        )
        return FakePyMaxMessage(id=333, chat_id=chat_id, sender=999, text=text)

    async def edit_message(self, chat_id, message_id, text, attachments=None):
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "text": text, "attachments": attachments})
        return FakePyMaxMessage(id=int(message_id), chat_id=chat_id, sender=999, text=text)


class FakePhotoAttachment:
    def __init__(self, base_url="https://cdn.example/photo.jpg"):
        self.base_url = base_url
        self.photo_id = 1
        self.type = "photo"


class FakeFileAttachment:
    def __init__(self, *, file_id=55, name="doc.pdf", token="tok"):
        self.file_id = file_id
        self.name = name
        self.token = token
        self.type = "file"


def test_register_exposes_max_userbot_platform_metadata():
    from plugins.platforms.max_userbot import register

    calls = []

    class Ctx:
        def register_platform(self, **kwargs):
            calls.append(kwargs)

    register(Ctx())

    assert len(calls) == 1
    entry = calls[0]
    assert entry["name"] == "max_userbot"
    assert entry["label"] == "MAX Userbot"
    assert entry["required_env"] == ["MAX_USERBOT_PHONE"]
    assert entry["allowed_users_env"] == "MAX_USERBOT_ALLOWED_USERS"
    assert entry["allow_all_env"] == "MAX_USERBOT_ALLOW_ALL_USERS"
    assert entry["max_message_length"] == 4000
    assert "internal MAX user account" in entry["platform_hint"]


def test_dynamic_platform_value_is_accepted_for_max_userbot_plugin():
    assert Platform("max_userbot").value == "max_userbot"


def test_session_work_dir_is_profile_local(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.max_userbot.adapter import MaxUserbotAdapter

    adapter = MaxUserbotAdapter(
        PlatformConfig(enabled=True, extra={"phone": "+79990000000", "session_name": "main.db"})
    )

    assert adapter.work_dir == str(tmp_path / "max_userbot")
    assert adapter.session_name == "main.db"


def test_validate_config_accepts_phone_or_existing_session(tmp_path, monkeypatch):
    monkeypatch.delenv("MAX_USERBOT_PHONE", raising=False)
    from plugins.platforms.max_userbot.adapter import validate_max_userbot_config

    assert validate_max_userbot_config(PlatformConfig(enabled=True)) is False
    assert validate_max_userbot_config(PlatformConfig(enabled=True, extra={"phone": "+799****0000"})) is True

    work_dir = tmp_path / "sessions"
    work_dir.mkdir()
    (work_dir / "main.db").write_text("sqlite-ish")
    assert validate_max_userbot_config(
        PlatformConfig(enabled=True, extra={"work_dir": str(work_dir), "session_name": "main.db"})
    ) is True

    monkeypatch.setenv("MAX_USERBOT_PHONE", "+799****0000")
    assert validate_max_userbot_config(PlatformConfig(enabled=True)) is True


def test_session_lock_prevents_two_userbot_clients_for_same_session(tmp_path):
    from plugins.platforms.max_userbot.adapter import MaxUserbotAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={"phone": "+799****0000", "work_dir": str(tmp_path), "session_name": "main.db"},
    )
    first = MaxUserbotAdapter(cfg)
    second = MaxUserbotAdapter(cfg)

    first._acquire_session_lock()
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            second._acquire_session_lock()
    finally:
        first._release_session_lock()

    second._acquire_session_lock()
    second._release_session_lock()


def test_message_to_event_maps_text_and_identity():
    from plugins.platforms.max_userbot.adapter import MaxUserbotAdapter

    adapter = MaxUserbotAdapter(PlatformConfig(enabled=True, extra={"phone": "+79990000000", "allow_all_users": True}))
    message = FakePyMaxMessage(id=111, chat_id=777, sender=123, text="hello")

    event = adapter._message_to_event(message)

    assert event is not None
    assert event.source.platform == Platform("max_userbot")
    assert event.source.chat_id == "777"
    assert event.source.user_id == "123"
    assert event.text == "hello"
    assert event.message_id == "111"
    assert event.message_type.value == "text"


def test_message_to_event_rejects_unallowlisted_sender():
    from plugins.platforms.max_userbot.adapter import MaxUserbotAdapter

    adapter = MaxUserbotAdapter(PlatformConfig(enabled=True, extra={"phone": "+79990000000", "allowed_users": ["123"]}))

    assert adapter._message_to_event(FakePyMaxMessage(sender=999)) is None
    assert adapter._message_to_event(FakePyMaxMessage(sender=123)) is not None


def test_message_to_event_ignores_own_account_messages():
    from plugins.platforms.max_userbot.adapter import MaxUserbotAdapter

    adapter = MaxUserbotAdapter(
        PlatformConfig(enabled=True, extra={"phone": "+79990000000", "allow_all_users": True, "account_id": "123"})
    )

    assert adapter._message_to_event(FakePyMaxMessage(sender=123, text="self echo")) is None


def test_message_to_event_maps_photo_and_file_attachments():
    from plugins.platforms.max_userbot.adapter import MaxUserbotAdapter

    adapter = MaxUserbotAdapter(PlatformConfig(enabled=True, extra={"phone": "+79990000000", "allow_all_users": True}))
    message = FakePyMaxMessage(
        text="see files",
        attaches=[FakePhotoAttachment("https://cdn.example/photo.jpg"), FakeFileAttachment(name="doc.pdf")],
    )

    event = adapter._message_to_event(message)

    assert event is not None
    assert event.message_type.value == "photo"
    assert event.media_urls == ["https://cdn.example/photo.jpg", "max_userbot_file:777:111:55"]
    assert event.media_types == ["image/jpeg", "application/pdf"]


@pytest.mark.asyncio
async def test_send_text_uses_pymax_client_send_message():
    from plugins.platforms.max_userbot.adapter import MaxUserbotAdapter

    adapter = MaxUserbotAdapter(PlatformConfig(enabled=True, extra={"phone": "+79990000000", "allow_all_users": True}))
    client = FakePyMaxClient()
    adapter._client = client

    result = await adapter.send("777", "hello", metadata={"notify": False})

    assert isinstance(result, SendResult)
    assert result.success is True
    assert result.message_id == "333"
    assert client.sent_messages == [
        {"chat_id": 777, "text": "hello", "reply_to": None, "attachments": None, "notify": False}
    ]


@pytest.mark.asyncio
async def test_send_text_honors_reply_to():
    from plugins.platforms.max_userbot.adapter import MaxUserbotAdapter

    adapter = MaxUserbotAdapter(PlatformConfig(enabled=True, extra={"phone": "+79990000000", "allow_all_users": True}))
    client = FakePyMaxClient()
    adapter._client = client

    result = await adapter.send("777", "reply", reply_to="111")

    assert result.success is True
    assert client.sent_messages[0]["reply_to"] == 111


@pytest.mark.asyncio
async def test_edit_message_uses_pymax_edit_message():
    from plugins.platforms.max_userbot.adapter import MaxUserbotAdapter

    adapter = MaxUserbotAdapter(PlatformConfig(enabled=True, extra={"phone": "+79990000000", "allow_all_users": True}))
    client = FakePyMaxClient()
    adapter._client = client

    result = await adapter.edit_message("777", "111", "updated", finalize=True)

    assert result.success is True
    assert result.message_id == "111"
    assert client.edits == [{"chat_id": 777, "message_id": 111, "text": "updated", "attachments": None}]


@pytest.mark.asyncio
async def test_inbound_handler_marks_read_and_dispatches(monkeypatch):
    from plugins.platforms.max_userbot.adapter import MaxUserbotAdapter

    adapter = MaxUserbotAdapter(
        PlatformConfig(enabled=True, extra={"phone": "+79990000000", "allow_all_users": True, "mark_read": True})
    )
    handled = []

    async def fake_handle(event):
        handled.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    message = FakePyMaxMessage(text="hello")

    await adapter._handle_pymax_message(message)

    assert message.read_called is True
    assert len(handled) == 1
    assert handled[0].text == "hello"


@pytest.mark.asyncio
async def test_send_exec_approval_uses_inline_keyboard_without_command_payload():
    from plugins.platforms.max_userbot.adapter import MaxUserbotAdapter

    adapter = MaxUserbotAdapter(PlatformConfig(enabled=True, extra={"phone": "+79990000000", "allow_all_users": True}))
    client = FakePyMaxClient()
    adapter._client = client

    result = await adapter.send_exec_approval(
        "777",
        command="rm -rf /tmp/secret",
        session_key="max_userbot:777",
        description="dangerous test command",
    )

    assert result.success is True
    assert adapter._approval_state == {1: "max_userbot:777"}
    sent = client.sent_messages[0]
    assert "Command Approval Required" in sent["text"]
    assert "rm -rf /tmp/secret" in sent["text"]
    keyboard = sent["attachments"][0]
    assert keyboard["type"] == "inline_keyboard"
    payloads = [button["payload"] for row in keyboard["payload"]["buttons"] for button in row]
    assert payloads == ["ea:once:1", "ea:session:1", "ea:always:1", "ea:deny:1"]
    assert all("rm -rf" not in payload for payload in payloads)
