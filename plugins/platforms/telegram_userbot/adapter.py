"""Telegram MTProto userbot platform plugin backed by Telethon.

This adapter is intentionally separate from ``plugins.platforms.telegram``.
The built-in Telegram platform uses the official Bot API; this plugin connects
as a normal Telegram user account over MTProto. User-account automation can be
restricted by Telegram and should be enabled only with explicit operator risk
acceptance and a strict allowlist.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import mimetypes
import os
import random
import tempfile
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from hermes_constants import get_hermes_home

Platform("telegram_userbot")

logger = logging.getLogger(__name__)


@dataclass
class _PacingTurn:
    """Mutable timing state shared only by tasks spawned for one inbound turn."""

    chat_id: str
    sender_id: str
    thinking_until: Optional[float] = None
    typing_started_at: Optional[float] = None
    typing_complete: bool = False
    typing_owner: object = field(default_factory=object, repr=False)


@dataclass(frozen=True)
class _TypingAction:
    """One owned Telethon typing action for a chat."""

    owner: object | None
    task: asyncio.Task


_current_pacing_turn: ContextVar[Optional[_PacingTurn]] = ContextVar(
    "telegram_userbot_current_pacing_turn", default=None
)

TELEGRAM_USERBOT_TEXT_LENGTH = 4096
DEFAULT_SESSION_NAME = "main"
DEFAULT_MAX_MEDIA_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_MEDIA_CACHE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_MEDIA_CACHE_FILES = 1000
_SECRET_EXTRA_KEYS = frozenset(
    {
        "api_id",
        "api_hash",
        "phone",
        "session_string",
        "two_factor_password",
        "2fa_password",
    }
)


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _split_csv(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, default: int) -> int:
    parsed = _coerce_int(value)
    return parsed if parsed is not None and parsed > 0 else default


def _bounded_ratio(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if 0.0 <= parsed <= 1.0 else default


def _first_present(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict):
            if name in obj and obj[name] is not None:
                return obj[name]
            continue
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _default_work_dir() -> str:
    return str(get_hermes_home() / "telegram_userbot")


def _normalized_session_name(value: Any) -> str:
    name = str(value or DEFAULT_SESSION_NAME).strip() or DEFAULT_SESSION_NAME
    return name[:-8] if name.endswith(".session") else name


def _reject_secret_config(extra: Dict[str, Any]) -> None:
    configured = sorted(key for key in _SECRET_EXTRA_KEYS if key in extra)
    if configured:
        raise ValueError(
            "Telegram userbot credentials must be stored in the profile .env, "
            f"not config extra: {', '.join(configured)}"
        )


def _safe_session_location(config: PlatformConfig) -> tuple[Path, str]:
    extra = config.extra or {}
    _reject_secret_config(extra)
    profile_root = get_hermes_home().expanduser().resolve()
    work_dir = Path(
        str(
            extra.get("work_dir")
            or os.getenv("TELEGRAM_USERBOT_WORK_DIR")
            or _default_work_dir()
        )
    ).expanduser().resolve()
    try:
        work_dir.relative_to(profile_root)
    except ValueError as exc:
        raise ValueError(
            f"Telegram userbot work_dir must stay under profile root {profile_root}"
        ) from exc

    session_name = _normalized_session_name(
        extra.get("session_name") or os.getenv("TELEGRAM_USERBOT_SESSION_NAME")
    )
    session_path = Path(session_name)
    if (
        session_path.is_absolute()
        or len(session_path.parts) != 1
        or session_name in {".", ".."}
    ):
        raise ValueError("Telegram userbot session_name must be a plain filename")
    return work_dir, session_name


def _session_file_from_config(config: PlatformConfig) -> Path:
    work_dir, session_name = _safe_session_location(config)
    return work_dir / f"{session_name}.session"


def check_telegram_userbot_requirements() -> bool:
    """Return True when Telethon is importable."""
    try:
        import telethon  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return False
    return True


def validate_telegram_userbot_config(config: PlatformConfig) -> bool:
    """Require env-only app credentials plus a bootstrap or saved session."""
    try:
        session_path = _session_file_from_config(config)
    except (TypeError, ValueError):
        return False
    api_id = _as_str(os.getenv("TELEGRAM_USERBOT_API_ID")).strip()
    api_hash = _as_str(os.getenv("TELEGRAM_USERBOT_API_HASH")).strip()
    if not api_id or not api_hash or _coerce_int(api_id) is None:
        return False
    phone = _as_str(os.getenv("TELEGRAM_USERBOT_PHONE")).strip()
    session_string = _as_str(
        os.getenv("TELEGRAM_USERBOT_SESSION_STRING")
    ).strip()
    return bool(phone or session_string or session_path.is_file())


def is_telegram_userbot_connected(config: PlatformConfig) -> bool:
    return validate_telegram_userbot_config(config)


class TelegramUserbotAdapter(BasePlatformAdapter):
    """Hermes gateway adapter backed by a Telegram MTProto user account."""

    MAX_MESSAGE_LENGTH = TELEGRAM_USERBOT_TEXT_LENGTH
    splits_long_messages = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("telegram_userbot"))
        extra = config.extra or {}
        work_dir, session_name = _safe_session_location(config)
        self.api_id = _coerce_int(os.getenv("TELEGRAM_USERBOT_API_ID"))
        self.api_hash = _as_str(os.getenv("TELEGRAM_USERBOT_API_HASH")).strip()
        self.phone = _as_str(os.getenv("TELEGRAM_USERBOT_PHONE")).strip()
        self.two_factor_password = _as_str(
            os.getenv("TELEGRAM_USERBOT_2FA_PASSWORD")
        )
        self.session_string = _as_str(
            os.getenv("TELEGRAM_USERBOT_SESSION_STRING")
        ).strip()
        self.work_dir = str(work_dir)
        self.session_name = session_name
        self.account_id = _as_str(
            extra.get("account_id") or os.getenv("TELEGRAM_USERBOT_ACCOUNT_ID")
        ).strip()
        self.allow_all_users = _coerce_bool(
            extra.get("allow_all_users")
            if "allow_all_users" in extra
            else os.getenv("TELEGRAM_USERBOT_ALLOW_ALL_USERS"),
            False,
        )
        self.allowed_users = set(
            _split_csv(
                extra.get("allowed_users")
                or extra.get("allow_from")
                or os.getenv("TELEGRAM_USERBOT_ALLOWED_USERS")
            )
        )
        self.allowed_chats = set(
            _split_csv(
                extra.get("allowed_chats")
                or os.getenv("TELEGRAM_USERBOT_ALLOWED_CHATS")
            )
        )
        # Every inbound event is gated by _message_to_event before dispatch.
        # Expose that local admission decision to the gateway so it does not
        # run the bot-oriented pairing flow a second time.
        self._dm_policy = "allowlist"
        self._group_policy = "allowlist"
        self.mark_read = _coerce_bool(extra.get("mark_read"), False)
        self.download_media = _coerce_bool(extra.get("download_media"), False)
        self.max_media_bytes = _positive_int(
            extra.get("max_media_bytes"), DEFAULT_MAX_MEDIA_BYTES
        )
        self.max_media_cache_bytes = _positive_int(
            extra.get("max_media_cache_bytes"), DEFAULT_MAX_MEDIA_CACHE_BYTES
        )
        self.max_media_cache_files = _positive_int(
            extra.get("max_media_cache_files"), DEFAULT_MAX_MEDIA_CACHE_FILES
        )
        self.send_typing_enabled = _coerce_bool(extra.get("send_typing"), True)
        self.human_pacing_enabled = _coerce_bool(
            extra.get("human_pacing_enabled"), False
        )
        self.human_pacing_excluded_user_ids = set(
            _split_csv(extra.get("human_pacing_excluded_user_ids"))
        )
        self.thinking_delay_min_ms = _positive_int(
            extra.get("thinking_delay_min_ms"), 1200
        )
        self.thinking_delay_max_ms = _positive_int(
            extra.get("thinking_delay_max_ms"), 3200
        )
        self.typing_chars_per_second = _positive_int(
            extra.get("typing_chars_per_second"), 12
        )
        self.typing_jitter_ratio = _bounded_ratio(
            extra.get("typing_jitter_ratio"), 0.15
        )
        self.typing_delay_min_ms = _positive_int(
            extra.get("typing_delay_min_ms"), 1800
        )
        self.typing_delay_max_ms = _positive_int(
            extra.get("typing_delay_max_ms"), 30000
        )
        self._client: Any = None
        self._client_task: Optional[asyncio.Task] = None
        self._typing_tasks: dict[str, _TypingAction] = {}
        self._typing_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._teardown_complete = asyncio.Event()
        self._teardown_complete.set()
        self._tearing_down = False
        self._media_lock = asyncio.Lock()
        self._session_lock_file: Optional[Any] = None

    @property
    def name(self) -> str:
        return "Telegram Userbot"

    @property
    def suppress_home_channel_onboarding(self) -> bool:
        """Keep user-account chats free of bot-oriented setup notices."""
        return True

    @property
    def suppress_system_messages(self) -> bool:
        """Keep Hermes command/error UI out of ordinary user-account chats."""
        return True

    @property
    def enforces_own_access_policy(self) -> bool:
        """Inbound senders are admitted locally before gateway dispatch."""
        return True

    @property
    def session_path(self) -> Path:
        return Path(self.work_dir).expanduser() / f"{self.session_name}.session"

    def _ensure_session_dir(self) -> None:
        path = Path(self.work_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass

    def _assert_safe_session_paths(self) -> None:
        lock_path = self.session_path.with_suffix(".session.lock")
        if self.session_path.is_symlink() or lock_path.is_symlink():
            raise RuntimeError(
                "Telegram userbot refuses symlink session or lock paths"
            )

    def _acquire_session_lock(self) -> None:
        if self.session_string or self._session_lock_file is not None:
            return
        self._ensure_session_dir()
        self._assert_safe_session_paths()
        lock_path = self.session_path.with_suffix(".session.lock")
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise RuntimeError(
                f"Unable to open Telegram userbot session lock safely: {lock_path}"
            ) from exc
        try:
            os.fchmod(descriptor, 0o600)
        except OSError:
            pass
        lock_file = os.fdopen(descriptor, "a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_file.close()
            if isinstance(exc, BlockingIOError):
                raise RuntimeError(
                    f"Telegram userbot session is already locked: {lock_path}"
                ) from exc
            raise RuntimeError(
                f"Unable to lock Telegram userbot session safely: {lock_path}"
            ) from exc
        self._session_lock_file = lock_file

    def _release_session_lock(self) -> None:
        lock_file = self._session_lock_file
        if lock_file is None:
            return
        self._session_lock_file = None
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def _build_client(self) -> Any:
        if not self.session_string:
            self._assert_safe_session_paths()
        try:
            from telethon import TelegramClient  # type: ignore[import-not-found]
            from telethon.sessions import StringSession  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - check_fn covers discovery
            raise RuntimeError(
                "Telethon is not installed; run `uv pip install 'telethon>=1.40,<2'`"
            ) from exc
        if not self.api_id or not self.api_hash:
            raise RuntimeError(
                "TELEGRAM_USERBOT_API_ID and TELEGRAM_USERBOT_API_HASH are required"
            )
        self._ensure_session_dir()
        session: Any
        if self.session_string:
            session = StringSession(self.session_string)
        else:
            session = str(Path(self.work_dir).expanduser() / self.session_name)
        return TelegramClient(session, self.api_id, self.api_hash)

    async def _await_critical_cleanup(self, cleanup_task: asyncio.Future) -> bool:
        cancelled = False
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                cancelled = True
        return cancelled

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        while True:
            await self._teardown_complete.wait()
            async with self._lifecycle_lock:
                if self._tearing_down:
                    continue
                if self._running:
                    return True
                return await self._connect_locked()

    async def _connect_locked(self) -> bool:
        try:
            self._acquire_session_lock()
            if self._client is None:
                self._client = self._build_client()
            from telethon import events  # type: ignore[import-not-found]

            self._client.add_event_handler(
                self._handle_new_message,
                events.NewMessage(incoming=True),
            )
            start_kwargs: dict[str, Any] = {}
            if self.phone:
                start_kwargs["phone"] = self.phone
            if self.two_factor_password:
                start_kwargs["password"] = self.two_factor_password
            await self._client.start(**start_kwargs)
            if not self.session_string and self.session_path.is_file():
                try:
                    os.chmod(self.session_path, 0o600)
                except OSError:
                    pass
            me = await self._client.get_me()
            if me is not None:
                self.account_id = _as_str(getattr(me, "id", None)).strip()
            self._mark_connected()
            self._client_task = asyncio.create_task(
                self._run_client_loop(self._client),
                name="telegram-userbot-client",
            )
            logger.warning(
                "[TELEGRAM_USERBOT] Started MTProto user-account client; "
                "this is not Telegram Bot API support."
            )
            return True
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self._cleanup_failed_connect())
            await self._await_critical_cleanup(cleanup)
            cleanup.result()
            raise
        except Exception as exc:
            logger.error("[TELEGRAM_USERBOT] failed to connect: %s", exc)
            cleanup = asyncio.create_task(self._cleanup_failed_connect())
            cancelled = await self._await_critical_cleanup(cleanup)
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError
            return False

    async def _disconnect_client_safely(
        self,
        client: Any,
        *,
        context: str,
    ) -> bool:
        if client is None:
            return False
        disconnect_task = asyncio.ensure_future(client.disconnect())
        cancelled = await self._await_critical_cleanup(disconnect_task)
        try:
            disconnect_task.result()
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            logger.debug("[TELEGRAM_USERBOT] %s disconnect failed: %s", context, exc)
        return cancelled

    async def _cleanup_failed_connect(self) -> None:
        self._mark_disconnected()
        client = self._client
        self._client = None
        await self._cancel_typing_tasks()
        await self._disconnect_client_safely(client, context="failed connect cleanup")
        task = self._client_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._client_task = None
        self._release_session_lock()

    async def _run_client_loop(self, client: Any) -> None:
        current_task = asyncio.current_task()
        try:
            await client.run_until_disconnected()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[TELEGRAM_USERBOT] client loop stopped: %s", exc)
        finally:
            self._tearing_down = True
            self._teardown_complete.clear()
            self._mark_disconnected()
            cleanup_task = asyncio.create_task(
                self._cleanup_client_loop(client, current_task),
                name="telegram-userbot-client-cleanup",
            )
            cancelled = await self._await_critical_cleanup(cleanup_task)
            disconnect_cancelled = cleanup_task.result()
            if cancelled or disconnect_cancelled:
                raise asyncio.CancelledError

    async def _cleanup_client_loop(
        self,
        client: Any,
        owner_task: Optional[asyncio.Task],
    ) -> bool:
        """Finish client-loop teardown despite repeated owner cancellation."""
        disconnect_cancelled = False
        try:
            await self._cancel_typing_tasks()
            disconnect_cancelled = await self._disconnect_client_safely(
                client,
                context="client loop cleanup",
            )
        finally:
            if self._client is client:
                self._client = None
            if self._client_task is owner_task:
                self._client_task = None
            self._release_session_lock()
            self._tearing_down = False
            self._teardown_complete.set()
        return disconnect_cancelled

    async def _cancel_typing_tasks(self) -> None:
        async with self._typing_lock:
            tasks = [action.task for action in self._typing_tasks.values()]
            self._typing_tasks.clear()
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def disconnect(self) -> None:
        while True:
            await self._teardown_complete.wait()
            async with self._lifecycle_lock:
                if self._tearing_down:
                    continue
                cleanup_task = asyncio.create_task(
                    self._disconnect_locked(),
                    name="telegram-userbot-disconnect-cleanup",
                )
                cancelled = await self._await_critical_cleanup(cleanup_task)
                cleanup_task.result()
                if cancelled:
                    raise asyncio.CancelledError
                return

    async def _disconnect_locked(self) -> None:
        self._mark_disconnected()
        client = self._client
        task = self._client_task
        await self._cancel_typing_tasks()
        disconnect_cancelled = await self._disconnect_client_safely(
            client,
            context="explicit",
        )
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._client_task = None
        self._client = None
        self._release_session_lock()
        if disconnect_cancelled:
            raise asyncio.CancelledError

    def _require_client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not content:
            return SendResult(
                success=False,
                error="Telegram userbot text message is empty",
            )
        pacing_turn = self._pacing_turn_for_chat(chat_id)
        try:
            await self._apply_human_pacing(chat_id, content)
            last: Optional[SendResult] = None
            for chunk in self.truncate_message(content, self.MAX_MESSAGE_LENGTH):
                try:
                    message = await self._require_client().send_message(
                        self._target_entity(chat_id),
                        chunk,
                        reply_to=_coerce_int(reply_to),
                    )
                except Exception as exc:
                    return SendResult(success=False, error=str(exc), retryable=True)
                last = SendResult(
                    success=True,
                    message_id=_as_str(getattr(message, "id", None)) or None,
                    raw_response=message,
                )
            return last or SendResult(success=False, error="No message chunks were sent")
        finally:
            if pacing_turn is not None:
                pacing_turn.typing_complete = True
                await self._stop_typing_now(
                    chat_id,
                    owner=pacing_turn.typing_owner,
                )

    def _pacing_turn_for_chat(self, chat_id: str) -> Optional[_PacingTurn]:
        turn = _current_pacing_turn.get()
        if (
            turn is None
            or turn.chat_id != str(chat_id)
            or not self._human_pacing_applies()
            or turn.thinking_until is None
        ):
            return None
        return turn

    async def _apply_human_pacing(self, chat_id: str, content: str) -> None:
        """Pad a fast response until its visible typing time looks plausible."""
        turn = _current_pacing_turn.get()
        if (
            not self._human_pacing_applies()
            or turn is None
            or turn.chat_id != str(chat_id)
        ):
            return
        thinking_until = turn.thinking_until
        if thinking_until is None:
            # Do not delay one-off outbound/system messages.
            return

        loop = asyncio.get_running_loop()
        remaining_thought = thinking_until - loop.time()
        if remaining_thought > 0:
            await asyncio.sleep(remaining_thought)

        typing_started = turn.typing_started_at
        if typing_started is None:
            typing_started = loop.time()
            turn.typing_started_at = typing_started
            await self.send_typing(chat_id)

        min_seconds = self.typing_delay_min_ms / 1000.0
        max_seconds = self.typing_delay_max_ms / 1000.0
        if min_seconds > max_seconds:
            min_seconds, max_seconds = max_seconds, min_seconds
        target_seconds = len(content) / float(self.typing_chars_per_second)
        target_seconds *= random.uniform(
            1.0 - self.typing_jitter_ratio,
            1.0 + self.typing_jitter_ratio,
        )
        target_seconds = max(min_seconds, min(max_seconds, target_seconds))
        remaining_typing = target_seconds - (loop.time() - typing_started)
        if remaining_typing > 0:
            await asyncio.sleep(remaining_typing)

    def _human_pacing_applies(self) -> bool:
        turn = _current_pacing_turn.get()
        return self.human_pacing_enabled and turn is not None and (
            turn.sender_id not in self.human_pacing_excluded_user_ids
        )

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        if not message_id:
            return SendResult(success=False, error="message_id required")
        if not content:
            return SendResult(
                success=False,
                error="Telegram userbot text message is empty",
            )
        try:
            message = await self._require_client().edit_message(
                self._target_entity(chat_id),
                _coerce_int(message_id) or message_id,
                content[: self.MAX_MESSAGE_LENGTH],
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        return SendResult(
            success=True,
            message_id=_as_str(getattr(message, "id", None) or message_id),
            raw_response=message,
        )

    async def _hold_typing(
        self,
        key: str,
        entity: Any,
        client: Any,
        owner: object | None,
    ) -> None:
        current_task = asyncio.current_task()
        try:
            async with client.action(entity, "typing"):
                await asyncio.sleep(4.0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "[TELEGRAM_USERBOT] typing action failed for chat %s: %s",
                key,
                exc,
            )
        finally:
            action = self._typing_tasks.get(key)
            if (
                action is not None
                and action.owner is owner
                and action.task is current_task
            ):
                self._typing_tasks.pop(key, None)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if not self.send_typing_enabled or self._client is None:
            return
        key = str(chat_id)
        turn = _current_pacing_turn.get()
        owner = turn.typing_owner if turn is not None and turn.chat_id == key else None
        if turn is not None and turn.chat_id == key and turn.typing_complete:
            return
        async with self._typing_lock:
            previous = self._typing_tasks.pop(key, None)
            if previous is not None:
                previous.task.cancel()
                await asyncio.gather(previous.task, return_exceptions=True)
            task = asyncio.create_task(
                self._hold_typing(
                    key,
                    self._target_entity(chat_id),
                    self._client,
                    owner,
                ),
                name=f"telegram-userbot-typing-{key}",
            )
            self._typing_tasks[key] = _TypingAction(owner=owner, task=task)
            await asyncio.sleep(0)

    async def stop_typing(self, chat_id: str) -> None:
        if self._pacing_turn_for_chat(chat_id) is not None:
            # Gateway asks to stop when generation finishes, before Base sends
            # the final response. Keep the action alive through the remaining
            # simulated typing delay; send() performs the immediate cleanup.
            return
        turn = _current_pacing_turn.get()
        owner = (
            turn.typing_owner
            if turn is not None and turn.chat_id == str(chat_id)
            else None
        )
        await self._stop_typing_now(chat_id, owner=owner)

    async def _stop_typing_now(
        self,
        chat_id: str,
        *,
        owner: object | None = None,
    ) -> None:
        key = str(chat_id)
        async with self._typing_lock:
            action = self._typing_tasks.get(key)
            if action is None or (owner is not None and action.owner is not owner):
                return
            self._typing_tasks.pop(key, None)
            action.task.cancel()
            await asyncio.gather(action.task, return_exceptions=True)

    async def _keep_typing(
        self,
        chat_id: str,
        interval: float = 2.0,
        metadata=None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        if not self._human_pacing_applies():
            await super()._keep_typing(
                chat_id,
                interval=interval,
                metadata=metadata,
                stop_event=stop_event,
            )
            return

        turn = _current_pacing_turn.get()
        if turn is None or turn.chat_id != str(chat_id):
            return
        loop = asyncio.get_running_loop()
        min_seconds = self.thinking_delay_min_ms / 1000.0
        max_seconds = self.thinking_delay_max_ms / 1000.0
        if min_seconds > max_seconds:
            min_seconds, max_seconds = max_seconds, min_seconds
        thinking_delay = random.uniform(min_seconds, max_seconds)
        thinking_until = loop.time() + thinking_delay
        turn.thinking_until = thinking_until
        try:
            await asyncio.sleep(thinking_delay)
            if stop_event is not None and stop_event.is_set():
                return
            turn.typing_started_at = loop.time()
            await super()._keep_typing(
                chat_id,
                interval=interval,
                metadata=metadata,
                stop_event=stop_event,
            )
        finally:
            if turn.thinking_until == thinking_until:
                turn.thinking_until = None
                turn.typing_started_at = None
                turn.typing_complete = True
            await self._stop_typing_now(chat_id, owner=turn.typing_owner)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        target = str(chat_id).strip()
        return {"id": target, "name": target, "type": "chat"}

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file(chat_id, image_path, caption, reply_to)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file(chat_id, file_path, caption, reply_to)

    async def send_video_file(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file(chat_id, video_path, caption, reply_to)

    async def _send_file(
        self,
        chat_id: str,
        path: str,
        caption: Optional[str],
        reply_to: Optional[str],
    ) -> SendResult:
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            return SendResult(
                success=False,
                error=f"Telegram userbot file does not exist: {path}",
            )
        try:
            message = await self._require_client().send_file(
                self._target_entity(chat_id),
                str(file_path),
                caption=caption or "",
                reply_to=_coerce_int(reply_to),
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        return SendResult(
            success=True,
            message_id=_as_str(getattr(message, "id", None)) or None,
            raw_response=message,
        )

    async def _process_message_background(
        self,
        event: MessageEvent,
        session_key: str,
    ) -> None:
        """Bind pacing identity to the event processed by this task.

        Base may drain a queued follow-up in a task created by the previous
        sender's turn. Rebinding here prevents that inherited ContextVar from
        applying the previous sender's pacing or bypass policy.
        """
        pacing_token = _current_pacing_turn.set(
            _PacingTurn(
                chat_id=str(event.source.chat_id),
                sender_id=_as_str(event.source.user_id).strip(),
            )
        )
        try:
            await super()._process_message_background(event, session_key)
        finally:
            _current_pacing_turn.reset(pacing_token)

    async def _handle_new_message(self, raw_event: Any) -> None:
        event = self._message_to_event(raw_event)
        if event is None:
            return
        if self.download_media and getattr(raw_event.message, "media", None) is not None:
            await self._download_event_media(raw_event, event)
        if self.mark_read:
            try:
                await self._require_client().send_read_acknowledge(raw_event.message)
            except Exception as exc:
                logger.debug(
                    "[TELEGRAM_USERBOT] mark_read failed for message %s: %s",
                    event.message_id,
                    exc,
                )
        pacing_token = _current_pacing_turn.set(
            _PacingTurn(
                chat_id=str(event.source.chat_id),
                sender_id=_as_str(event.source.user_id).strip(),
            )
        )
        try:
            await self.handle_message(event)
        finally:
            _current_pacing_turn.reset(pacing_token)

    def _message_to_event(self, raw_event: Any) -> Optional[MessageEvent]:
        message = getattr(raw_event, "message", raw_event)
        if _coerce_bool(getattr(message, "out", False), False):
            return None
        chat_id = _as_str(
            _first_present(raw_event, "chat_id")
            or _first_present(message, "chat_id", "peer_id")
        ).strip()
        user_id = _as_str(
            _first_present(raw_event, "sender_id")
            or _first_present(message, "sender_id", "from_id")
        ).strip()
        if not chat_id or self._is_own_user(user_id):
            return None
        if not self._is_authorized(user_id, chat_id):
            return None
        text = _as_str(
            _first_present(raw_event, "raw_text")
            or _first_present(message, "raw_text", "message", "text")
        )
        has_media = getattr(message, "media", None) is not None
        if not text.strip() and not has_media:
            return None
        source = self._source_for_event(raw_event, chat_id, user_id)
        return MessageEvent(
            text=text,
            message_type=self._message_type(message),
            source=source,
            raw_message=raw_event,
            message_id=_as_str(_first_present(message, "id", "message_id")) or None,
            timestamp=self._timestamp(message),
        )

    def _source_for_event(self, raw_event: Any, chat_id: str, user_id: str):
        from gateway.session import SessionSource

        is_private = bool(getattr(raw_event, "is_private", False))
        return SessionSource(
            platform=Platform("telegram_userbot"),
            chat_id=chat_id,
            chat_type="dm" if is_private else "group",
            user_id=user_id or None,
        )

    @staticmethod
    def _media_size(message: Any) -> Optional[int]:
        candidates: list[Any] = []
        file_info = getattr(message, "file", None)
        candidates.append(getattr(file_info, "size", None))
        document = getattr(message, "document", None)
        candidates.append(getattr(document, "size", None))
        photo = getattr(message, "photo", None)
        for size_info in getattr(photo, "sizes", None) or []:
            candidates.append(getattr(size_info, "size", None))
        sizes = [
            parsed
            for value in candidates
            if (parsed := _coerce_int(value)) is not None and parsed >= 0
        ]
        return max(sizes) if sizes else None

    @staticmethod
    def _media_cache_usage(media_dir: Path) -> tuple[int, int]:
        files = 0
        total = 0
        for path in media_dir.rglob("*"):
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                files += 1
                total += path.stat().st_size
            except OSError:
                continue
        return files, total

    @staticmethod
    def _media_suffix(message: Any) -> str:
        file_info = getattr(message, "file", None)
        original_name = _as_str(getattr(file_info, "name", None)).strip()
        suffix = Path(original_name).suffix.lower()
        if suffix and len(suffix) <= 10 and suffix[1:].isalnum():
            return suffix
        document = getattr(message, "document", None)
        mime = _as_str(getattr(document, "mime_type", None)).strip().lower()
        guessed = mimetypes.guess_extension(mime) or ""
        return guessed if len(guessed) <= 10 else ""

    async def _download_event_media(
        self,
        raw_event: Any,
        event: MessageEvent,
    ) -> None:
        message = raw_event.message
        expected_size = self._media_size(message)
        if expected_size is None:
            logger.warning(
                "[TELEGRAM_USERBOT] refusing media with unknown size for message %s",
                event.message_id,
            )
            return
        if expected_size > self.max_media_bytes:
            logger.warning(
                "[TELEGRAM_USERBOT] refusing oversized media for message %s (%s > %s)",
                event.message_id,
                expected_size,
                self.max_media_bytes,
            )
            return

        async with self._media_lock:
            downloaded = await self._download_media_bounded(
                message,
                event.message_id,
                expected_size,
            )
        if downloaded is None:
            return

        local_path = str(downloaded)
        event.media_urls.append(local_path)
        document = getattr(message, "document", None)
        declared_mime = _as_str(getattr(document, "mime_type", None)).strip()
        mime = declared_mime or mimetypes.guess_type(local_path)[0]
        mime = mime or "application/octet-stream"
        event.media_types.append(mime)
        if mime.startswith("image/"):
            event.message_type = MessageType.PHOTO
        elif mime.startswith("audio/"):
            event.message_type = MessageType.AUDIO
        elif mime.startswith("video/"):
            event.message_type = MessageType.VIDEO
        else:
            event.message_type = MessageType.DOCUMENT

    async def _download_media_bounded(
        self,
        message: Any,
        message_id: Optional[str],
        expected_size: int,
    ) -> Optional[Path]:
        media_path = Path(self.work_dir).expanduser() / "media"
        if media_path.is_symlink():
            logger.error(
                "[TELEGRAM_USERBOT] refusing symlink media cache: %s",
                media_path,
            )
            return None
        media_path.mkdir(parents=True, exist_ok=True)
        media_dir = media_path.resolve()
        try:
            media_dir.relative_to(Path(self.work_dir).expanduser().resolve())
        except ValueError:
            logger.error(
                "[TELEGRAM_USERBOT] refusing media cache outside profile work dir: %s",
                media_dir,
            )
            return None
        try:
            os.chmod(media_dir, 0o700)
        except OSError:
            pass

        file_count, cache_bytes = self._media_cache_usage(media_dir)
        if (
            file_count >= self.max_media_cache_files
            or cache_bytes + expected_size > self.max_media_cache_bytes
        ):
            logger.warning(
                "[TELEGRAM_USERBOT] media cache quota reached; refusing message %s",
                message_id,
            )
            return None

        descriptor, raw_path = tempfile.mkstemp(
            prefix="incoming-",
            suffix=self._media_suffix(message),
            dir=media_dir,
        )
        downloaded = Path(raw_path)
        actual_size = 0
        exceeded = False
        try:
            with os.fdopen(descriptor, "wb") as output:
                async for chunk in self._require_client().iter_download(
                    message.media,
                    request_size=512 * 1024,
                ):
                    payload = bytes(chunk)
                    next_size = actual_size + len(payload)
                    if (
                        next_size > self.max_media_bytes
                        or cache_bytes + next_size > self.max_media_cache_bytes
                    ):
                        exceeded = True
                        break
                    output.write(payload)
                    actual_size = next_size
        except asyncio.CancelledError:
            try:
                downloaded.unlink()
            except OSError:
                pass
            raise
        except Exception as exc:
            try:
                downloaded.unlink()
            except OSError:
                pass
            logger.warning(
                "[TELEGRAM_USERBOT] media download failed for message %s: %s",
                message_id,
                exc,
            )
            return None

        final_files, final_bytes = self._media_cache_usage(media_dir)
        if (
            exceeded
            or final_files > self.max_media_cache_files
            or final_bytes > self.max_media_cache_bytes
        ):
            try:
                downloaded.unlink()
            except OSError:
                pass
            logger.warning(
                "[TELEGRAM_USERBOT] media exceeded quota for message %s",
                message_id,
            )
            return None
        return downloaded

    def _is_own_user(self, user_id: str) -> bool:
        return bool(user_id and self.account_id and user_id == self.account_id)

    def _is_authorized(self, user_id: str, chat_id: str) -> bool:
        if self.allow_all_users or _coerce_bool(
            os.getenv("GATEWAY_ALLOW_ALL_USERS"), False
        ):
            return True
        allowed_users = set(self.allowed_users)
        allowed_users.update(_split_csv(os.getenv("GATEWAY_ALLOWED_USERS")))
        if user_id and (user_id in allowed_users or "*" in allowed_users):
            return True
        if chat_id and (
            chat_id in self.allowed_chats or "*" in self.allowed_chats
        ):
            return True
        return False

    @staticmethod
    def _message_type(message: Any) -> MessageType:
        if getattr(message, "photo", None) is not None:
            return MessageType.PHOTO
        document = getattr(message, "document", None)
        mime = _as_str(getattr(document, "mime_type", None)).lower()
        if mime.startswith("audio/"):
            return MessageType.AUDIO
        if mime.startswith("video/"):
            return MessageType.VIDEO
        if document is not None or getattr(message, "media", None) is not None:
            return MessageType.DOCUMENT
        return MessageType.TEXT

    @staticmethod
    def _timestamp(message: Any) -> datetime:
        value = getattr(message, "date", None)
        return value if isinstance(value, datetime) else datetime.now(timezone.utc)

    @staticmethod
    def _target_entity(value: Any) -> Any:
        raw = str(value).strip()
        if raw.startswith("chat:") or raw.startswith("user:"):
            raw = raw.split(":", 1)[1]
        parsed = _coerce_int(raw)
        return parsed if parsed is not None else raw


TELEGRAM_USERBOT_PLATFORM_HINT = (
    "You are on Telegram through an MTProto user account, not the Telegram "
    "Bot API. Keep responses concise and chat-friendly. Never reveal Telegram "
    "application credentials, login codes, two-factor passwords, or session "
    "strings."
)


def register(ctx) -> None:
    ctx.register_platform(
        name="telegram_userbot",
        label="Telegram Userbot",
        adapter_factory=lambda cfg: TelegramUserbotAdapter(cfg),
        check_fn=check_telegram_userbot_requirements,
        validate_config=validate_telegram_userbot_config,
        is_connected=is_telegram_userbot_connected,
        required_env=[
            "TELEGRAM_USERBOT_API_ID",
            "TELEGRAM_USERBOT_API_HASH",
        ],
        allowed_users_env="TELEGRAM_USERBOT_ALLOWED_USERS",
        allow_all_env="TELEGRAM_USERBOT_ALLOW_ALL_USERS",
        max_message_length=TELEGRAM_USERBOT_TEXT_LENGTH,
        platform_hint=TELEGRAM_USERBOT_PLATFORM_HINT,
        install_hint="Install Telethon with `uv pip install 'telethon>=1.40,<2'` and bootstrap a profile-local session in the foreground.",
        emoji="👤",
    )
