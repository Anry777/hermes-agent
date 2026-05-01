import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig, load_gateway_config


def _make_adapter(require_mention=None, free_response_chats=None, mention_patterns=None, ignored_threads=None, reply_triggers=None):
    from gateway.platforms.telegram import TelegramAdapter

    extra = {}
    if require_mention is not None:
        extra["require_mention"] = require_mention
    if free_response_chats is not None:
        extra["free_response_chats"] = free_response_chats
    if mention_patterns is not None:
        extra["mention_patterns"] = mention_patterns
    if ignored_threads is not None:
        extra["ignored_threads"] = ignored_threads
    if reply_triggers is not None:
        extra["reply_triggers"] = reply_triggers

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._mention_patterns = adapter._compile_mention_patterns()
    return adapter


def _group_message(
    text="hello",
    *,
    chat_id=-100,
    thread_id=None,
    reply_to_bot=False,
    reply_to_user=None,
    entities=None,
    caption=None,
    caption_entities=None,
):
    reply_to_message = None
    if reply_to_bot:
        reply_to_message = SimpleNamespace(from_user=SimpleNamespace(id=999, is_bot=True))
    elif reply_to_user is not None:
        reply_to_message = SimpleNamespace(from_user=reply_to_user)
    return SimpleNamespace(
        text=text,
        caption=caption,
        entities=entities or [],
        caption_entities=caption_entities or [],
        message_thread_id=thread_id,
        chat=SimpleNamespace(id=chat_id, type="group"),
        reply_to_message=reply_to_message,
    )


def _mention_entity(text, mention="@hermes_bot"):
    offset = text.index(mention)
    return SimpleNamespace(type="mention", offset=offset, length=len(mention))


def _bot_command_entity(text, command):
    """Entity Telegram emits for a ``/cmd`` or ``/cmd@botname`` token.

    Telegram parses slash commands server-side. For ``/cmd@botname`` the
    client does NOT emit a separate ``mention`` entity — the whole span
    is a single ``bot_command`` entity.
    """
    offset = text.index(command)
    return SimpleNamespace(type="bot_command", offset=offset, length=len(command))


def test_group_messages_can_be_opened_via_config():
    adapter = _make_adapter(require_mention=False)

    assert adapter._should_process_message(_group_message("hello everyone")) is True




def test_group_reply_trigger_can_be_disabled_for_noisy_multi_bot_chats():
    adapter = _make_adapter(require_mention=True, reply_triggers=False)

    assert adapter._should_process_message(_group_message("replying", reply_to_bot=True)) is False
    assert adapter._should_process_message(
        _group_message("hi @hermes_bot", entities=[_mention_entity("hi @hermes_bot")])
    ) is True


def test_direct_only_without_reply_trigger_ignores_plain_replies_in_target_group():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=[],
        mention_patterns=[r"^\s*(Цезарь|Cesair)[,!:：]?\b"],
        reply_triggers=False,
    )

    assert adapter._should_process_message(
        _group_message("тебя не спрашивали", chat_id=-5283179051, reply_to_bot=True)
    ) is False
    assert adapter._should_process_message(
        _group_message("Цезарь, ответь", chat_id=-5283179051)
    ) is True

def test_group_messages_can_require_direct_trigger_via_config():
    adapter = _make_adapter(require_mention=True)

    assert adapter._should_process_message(_group_message("hello everyone")) is False
    assert adapter._should_process_message(_group_message("hi @hermes_bot", entities=[_mention_entity("hi @hermes_bot")])) is True
    assert adapter._should_process_message(_group_message("replying", reply_to_bot=True)) is True
    # Commands must also respect require_mention when it is enabled
    assert adapter._should_process_message(_group_message("/status"), is_command=True) is False
    # Telegram's group command menu sends ``/cmd@botname`` as a single
    # ``bot_command`` entity spanning the whole token (no separate mention
    # entity). We must accept it so the menu works when require_mention is on.
    assert adapter._should_process_message(
        _group_message(
            "/status@hermes_bot",
            entities=[_bot_command_entity("/status@hermes_bot", "/status@hermes_bot")],
        ),
        is_command=True,
    ) is True
    # A bot_command entity addressed at a different bot must not satisfy
    # the mention gate — Telegram groups can host multiple bots that
    # register the same command name.
    assert adapter._should_process_message(
        _group_message(
            "/status@other_bot",
            entities=[_bot_command_entity("/status@other_bot", "/status@other_bot")],
        ),
        is_command=True,
    ) is False
    # Bare ``/status`` (no @botname) must still be dropped in groups with
    # require_mention=True — Telegram delivers it only when the bot's
    # privacy mode is off, and even then we should not respond unless the
    # user explicitly addressed the bot.
    assert adapter._should_process_message(
        _group_message("/status", entities=[_bot_command_entity("/status", "/status")]),
        is_command=True,
    ) is False
    # And commands still pass unconditionally when require_mention is disabled
    adapter_no_mention = _make_adapter(require_mention=False)
    assert adapter_no_mention._should_process_message(_group_message("/status"), is_command=True) is True


def test_free_response_chats_allow_ambient_non_questions_without_hijacking_questions():
    adapter = _make_adapter(require_mention=True, free_response_chats=["-200"])

    assert adapter._should_process_message(_group_message("обычный разговор без вопроса", chat_id=-200)) is True
    assert adapter._should_process_message(_group_message("кто я?", chat_id=-200)) is False
    assert adapter._should_process_message(_group_message("hello everyone", chat_id=-201)) is False



def test_free_response_chats_ignore_messages_addressed_to_other_bot():
    adapter = _make_adapter(require_mention=True, free_response_chats=["-200"])

    other_text = "@other_bot посмотри"
    assert adapter._should_process_message(
        _group_message(
            other_text,
            chat_id=-200,
            entities=[_mention_entity(other_text, "@other_bot")],
        )
    ) is False

    other_command = "/status@other_bot"
    assert adapter._should_process_message(
        _group_message(
            other_command,
            chat_id=-200,
            entities=[_bot_command_entity(other_command, other_command)],
        ),
        is_command=True,
    ) is False




def test_explicit_other_mention_beats_local_wake_word_and_reply_context():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=["-200"],
        mention_patterns=[r"^\s*(Цезарь|Cesair)[,!:：]?\b"],
    )

    text = "@DashaHermesBot Цезарь, вот и сиди"
    assert adapter._should_process_message(
        _group_message(
            text,
            chat_id=-200,
            reply_to_bot=True,
            entities=[_mention_entity(text, "@DashaHermesBot")],
        )
    ) is False

def test_free_response_chats_still_accept_direct_mentions_and_replies():
    adapter = _make_adapter(require_mention=True, free_response_chats=["-200"])

    text = "@hermes_bot посмотри"
    assert adapter._should_process_message(
        _group_message(text, chat_id=-200, entities=[_mention_entity(text)])
    ) is True
    assert adapter._should_process_message(_group_message("обычный reply", chat_id=-200, reply_to_bot=True)) is True


def test_reply_context_does_not_override_fresh_addressing_to_other_bot():
    adapter = _make_adapter(require_mention=True, free_response_chats=["-200"])

    other_mention = "@other_bot посмотри"
    assert adapter._should_process_message(
        _group_message(
            other_mention,
            chat_id=-200,
            reply_to_bot=True,
            entities=[_mention_entity(other_mention, "@other_bot")],
        )
    ) is False

    assert adapter._should_process_message(
        _group_message("Борис, скажи кто я", chat_id=-200, reply_to_bot=True)
    ) is False


def test_free_response_chats_ignore_replies_to_other_bots():
    adapter = _make_adapter(require_mention=True, free_response_chats=["-200"])

    assert adapter._should_process_message(
        _group_message(
            "да, посмотри",
            chat_id=-200,
            reply_to_user=SimpleNamespace(id=12345, is_bot=True),
        )
    ) is False
    assert adapter._should_process_message(
        _group_message(
            "обычный reply человеку",
            chat_id=-200,
            reply_to_user=SimpleNamespace(id=54321, is_bot=False),
        )
    ) is True


def test_free_response_chats_ignore_generic_leading_vocative_without_name_blacklist():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=["-200"],
        mention_patterns=[r"^\s*(Федя|Фёдор|Федор)[,:]?\b"],
    )

    assert adapter._should_process_message(_group_message("Борис, скажи кто я", chat_id=-200)) is False
    assert adapter._should_process_message(_group_message("Маша, как дела?", chat_id=-200)) is False
    assert adapter._should_process_message(_group_message("обычное сообщение", chat_id=-200)) is True
    assert adapter._should_process_message(_group_message("Федя, ты тут?", chat_id=-200)) is True
    assert adapter._should_process_message(
        _group_message("@hermes_bot Борис тут?", chat_id=-200, entities=[_mention_entity("@hermes_bot Борис тут?")])
    ) is True


def test_ignored_threads_drop_group_messages_before_other_gates():
    adapter = _make_adapter(require_mention=False, free_response_chats=["-200"], ignored_threads=[31, "42"])

    assert adapter._should_process_message(_group_message("hello everyone", chat_id=-200, thread_id=31)) is False
    assert adapter._should_process_message(_group_message("hello everyone", chat_id=-200, thread_id=42)) is False
    assert adapter._should_process_message(_group_message("hello everyone", chat_id=-200, thread_id=99)) is True


def test_regex_mention_patterns_allow_custom_wake_words():
    adapter = _make_adapter(require_mention=True, mention_patterns=[r"^\s*chompy\b"])

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("   chompy help")) is True
    assert adapter._should_process_message(_group_message("hey chompy")) is False


def test_invalid_regex_patterns_are_ignored():
    adapter = _make_adapter(require_mention=True, mention_patterns=[r"(", r"^\s*chompy\b"])

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("hello everyone")) is False


def test_config_bridges_telegram_group_settings(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n"
        "  require_mention: true\n"
        "  reply_triggers: false\n"
        "  mention_patterns:\n"
        "    - \"^\\\\s*chompy\\\\b\"\n"
        "  free_response_chats:\n"
        "    - \"-123\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("TELEGRAM_MENTION_PATTERNS", raising=False)
    monkeypatch.delenv("TELEGRAM_FREE_RESPONSE_CHATS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert __import__("os").environ["TELEGRAM_REQUIRE_MENTION"] == "true"
    assert json.loads(__import__("os").environ["TELEGRAM_MENTION_PATTERNS"]) == [r"^\s*chompy\b"]
    assert __import__("os").environ["TELEGRAM_FREE_RESPONSE_CHATS"] == "-123"


def test_config_bridges_telegram_ignored_threads(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n"
        "  ignored_threads:\n"
        "    - 31\n"
        "    - \"42\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_IGNORED_THREADS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert __import__("os").environ["TELEGRAM_IGNORED_THREADS"] == "31,42"
