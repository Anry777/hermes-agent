"""MAX userbot platform plugin adapter.

This adapter is intentionally separate from ``plugins.platforms.max``.  The
``max`` plugin uses the official MAX Bot API and a bot token; this plugin uses
MaxApiTeam/PyMax (``maxapi-python``) to connect as a normal user account via
MAX internal APIs.  It is therefore useful for home/local deployments where a
second official bot cannot be created, but it is unofficial and can break when
MAX changes its private API.
"""

from __future__ import annotations

import asyncio
import fcntl
import itertools
import logging
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_url,
    safe_url_for_log,
)
from hermes_constants import get_hermes_home

# Ensure the plugin platform pseudo-member is registered once this bundled
# plugin exists on disk.  See gateway.config.Platform._missing_().
Platform("max_userbot")

logger = logging.getLogger(__name__)

MAX_USERBOT_TEXT_LENGTH = 4000
DEFAULT_SESSION_NAME = "main.db"
DEFAULT_CLIENT_TYPE = "tcp"
SUPPORTED_CLIENT_TYPES = {"tcp", "web"}


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


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


def _default_work_dir() -> str:
    return str(get_hermes_home() / "max_userbot")


def _session_file_from_config(config: PlatformConfig) -> Path:
    extra = config.extra or {}
    work_dir = str(extra.get("work_dir") or os.getenv("MAX_USERBOT_WORK_DIR") or _default_work_dir())
    session_name = str(extra.get("session_name") or os.getenv("MAX_USERBOT_SESSION_NAME") or DEFAULT_SESSION_NAME)
    return Path(work_dir).expanduser() / session_name


def check_max_userbot_requirements() -> bool:
    """Return True when the optional PyMax dependency is importable."""
    try:
        import pymax  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return False
    return True


def validate_max_userbot_config(config: PlatformConfig) -> bool:
    """Accept either first-time phone auth or an existing profile-local session."""
    extra = config.extra or {}
    phone = str(extra.get("phone") or os.getenv("MAX_USERBOT_PHONE") or "").strip()
    if phone:
        return True
    try:
        return _session_file_from_config(config).exists()
    except Exception:
        return False


def is_max_userbot_connected(config: PlatformConfig) -> bool:
    return validate_max_userbot_config(config)


class _EnvPasswordProvider:
    async def get_password(self, hint: str | None = None) -> str:
        return os.environ.get("MAX_USERBOT_2FA_PASSWORD", "")


class MaxUserbotAdapter(BasePlatformAdapter):
    """Hermes gateway adapter backed by a PyMax user account client."""

    MAX_MESSAGE_LENGTH = MAX_USERBOT_TEXT_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("max_userbot"))
        extra = config.extra or {}
        self.phone = str(extra.get("phone") or os.getenv("MAX_USERBOT_PHONE") or "").strip()
        client_type = str(extra.get("client_type") or os.getenv("MAX_USERBOT_CLIENT_TYPE") or DEFAULT_CLIENT_TYPE).strip().lower()
        self.client_type = client_type if client_type in SUPPORTED_CLIENT_TYPES else DEFAULT_CLIENT_TYPE
        self.work_dir = str(extra.get("work_dir") or os.getenv("MAX_USERBOT_WORK_DIR") or _default_work_dir())
        self.session_name = str(extra.get("session_name") or os.getenv("MAX_USERBOT_SESSION_NAME") or DEFAULT_SESSION_NAME)
        self.account_id = _as_str(extra.get("account_id") or os.getenv("MAX_USERBOT_ACCOUNT_ID")).strip()
        self.allow_all_users = _coerce_bool(
            extra.get("allow_all_users") if "allow_all_users" in extra else os.getenv("MAX_USERBOT_ALLOW_ALL_USERS"),
            False,
        )
        self.allowed_users = set(_split_csv(extra.get("allowed_users") or extra.get("allow_from") or os.getenv("MAX_USERBOT_ALLOWED_USERS")))
        self.allowed_chats = set(_split_csv(extra.get("allowed_chats") or os.getenv("MAX_USERBOT_ALLOWED_CHATS")))
        self.mark_read = _coerce_bool(
            extra.get("mark_read") if "mark_read" in extra else os.getenv("MAX_USERBOT_MARK_READ"),
            False,
        )
        self.download_media = _coerce_bool(
            extra.get("download_media") if "download_media" in extra else os.getenv("MAX_USERBOT_DOWNLOAD_MEDIA"),
            True,
        )
        self.send_typing_enabled = _coerce_bool(
            extra.get("send_typing") if "send_typing" in extra else os.getenv("MAX_USERBOT_SEND_TYPING"),
            True,
        )
        self._client: Any = None
        self._client_task: Optional[asyncio.Task] = None
        self._session_lock_file: Optional[Any] = None
        self._approval_counter = itertools.count(1)
        self._approval_state: Dict[int, str] = {}

    @property
    def name(self) -> str:
        return "MAX Userbot"

    @property
    def session_path(self) -> Path:
        return Path(self.work_dir).expanduser() / self.session_name

    def _ensure_session_dir(self) -> None:
        path = Path(self.work_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass

    def _acquire_session_lock(self) -> None:
        if self._session_lock_file is not None:
            return
        self._ensure_session_dir()
        lock_path = Path(self.work_dir).expanduser() / f"{self.session_name}.lock"
        lock_file = open(lock_path, "a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise RuntimeError(f"MAX userbot session is already locked: {lock_path}") from exc
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

    def _build_pymax_client(self) -> Any:
        try:
            from pymax import Client, WebClient  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - covered by check_fn path
            raise RuntimeError("maxapi-python is not installed; run `uv pip install maxapi-python`") from exc

        self._ensure_session_dir()
        if self.client_type == "web":
            return WebClient(work_dir=self.work_dir, session_name=self.session_name)
        if not self.phone:
            raise RuntimeError("MAX_USERBOT_PHONE is required for PyMax TCP Client first-time authorization")
        kwargs: dict[str, Any] = {
            "phone": self.phone,
            "work_dir": self.work_dir,
            "session_name": self.session_name,
        }
        if os.getenv("MAX_USERBOT_2FA_PASSWORD"):
            kwargs["password_provider"] = _EnvPasswordProvider()
        return Client(**kwargs)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if self._running:
            return True
        try:
            self._acquire_session_lock()
            if self._client is None:
                self._client = self._build_pymax_client()
            self._register_pymax_handlers(self._client)
            self._mark_connected()
            self._client_task = asyncio.create_task(self._client.start(), name="max-userbot-client")
            logger.warning(
                "[MAX_USERBOT] Started unofficial PyMax userbot client; this is not MAX Bot API support."
            )
            return True
        except Exception as exc:
            logger.error("[MAX_USERBOT] failed to connect: %s", exc)
            self._mark_disconnected()
            self._release_session_lock()
            return False

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._client_task is not None:
            self._client_task.cancel()
            await asyncio.gather(self._client_task, return_exceptions=True)
            self._client_task = None
        if self._client is not None:
            close = getattr(self._client, "close", None) or getattr(self._client, "stop", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            self._client = None
        self._release_session_lock()

    def _register_pymax_handlers(self, client: Any) -> None:
        if hasattr(client, "on_message"):
            client.on_message()(self._handle_pymax_message)
        if hasattr(client, "on_raw"):
            client.on_raw()(self._handle_pymax_raw)

    def _require_client(self) -> Any:
        if self._client is None:
            self._client = self._build_pymax_client()
        return self._client

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        metadata = metadata or {}
        if not content:
            return SendResult(success=False, error="MAX userbot text message is empty")
        client = self._require_client()
        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH)
        last: Optional[SendResult] = None
        for chunk in chunks:
            try:
                msg = await client.send_message(
                    chat_id=self._target_chat_id(chat_id),
                    text=chunk,
                    reply_to=_coerce_int(reply_to),
                    attachments=None,
                    notify=bool(metadata.get("notify", True)),
                )
            except TypeError:
                msg = await client.send_message(
                    chat_id=self._target_chat_id(chat_id),
                    text=chunk,
                    reply_to=_coerce_int(reply_to),
                )
            except Exception as exc:
                return SendResult(success=False, error=str(exc), retryable=True)
            last = SendResult(success=True, message_id=_as_str(_first_present(msg, "id", "message_id")), raw_response=msg)
        return last or SendResult(success=False, error="No MAX userbot message chunks were sent")

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
            return SendResult(success=False, error="MAX userbot text message is empty")
        client = self._require_client()
        try:
            msg = await client.edit_message(
                chat_id=self._target_chat_id(chat_id),
                message_id=self._target_chat_id(message_id),
                text=content[: self.MAX_MESSAGE_LENGTH],
                attachments=None,
            )
        except TypeError:
            try:
                msg = await client.edit_message(
                    chat_id=self._target_chat_id(chat_id),
                    message_id=self._target_chat_id(message_id),
                    text=content[: self.MAX_MESSAGE_LENGTH],
                )
            except Exception as exc:
                return SendResult(success=False, error=str(exc), retryable=True)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        return SendResult(success=True, message_id=_as_str(_first_present(msg, "id", "message_id") or message_id), raw_response=msg)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if not self.send_typing_enabled:
            return
        client = self._client
        if client is None:
            return
        method = getattr(client, "send_typing", None)
        if method is None:
            return
        try:
            result = method(chat_id=self._target_chat_id(chat_id))
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug("[MAX_USERBOT] send_typing failed for chat %s: %s", chat_id, exc)

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
        return await self._send_file_attachment(chat_id, image_path, caption, reply_to, notify=(metadata or {}).get("notify", True), kind="photo")

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
        return await self._send_file_attachment(chat_id, file_path, caption, reply_to, notify=(metadata or {}).get("notify", True), kind="file")

    async def send_video_file(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_attachment(chat_id, video_path, caption, reply_to, notify=(metadata or {}).get("notify", True), kind="video")

    async def _send_file_attachment(
        self,
        chat_id: str,
        path: str,
        caption: Optional[str],
        reply_to: Optional[str],
        *,
        notify: bool,
        kind: str,
    ) -> SendResult:
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            return SendResult(success=False, error=f"MAX userbot file does not exist: {path}")
        try:
            from pymax import File, Photo, Video  # type: ignore[import-not-found]
            cls = {"photo": Photo, "video": Video}.get(kind, File)
            attachment = cls(path=str(file_path))
            msg = await self._require_client().send_message(
                chat_id=self._target_chat_id(chat_id),
                text=caption or "",
                reply_to=_coerce_int(reply_to),
                attachments=[attachment],
                notify=notify,
            )
            return SendResult(success=True, message_id=_as_str(_first_present(msg, "id", "message_id")), raw_response=msg)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        approval_id = next(self._approval_counter)
        cmd_preview = command[:3200] + "..." if len(command) > 3200 else command
        desc_preview = description[:500] + "..." if len(description) > 500 else description
        text = (
            "⚠️ **Command Approval Required**\n\n"
            f"```\n{cmd_preview}\n```\n\n"
            f"Reason: {desc_preview}"
        )
        keyboard = {
            "type": "inline_keyboard",
            "payload": {
                "buttons": [
                    [
                        {"type": "callback", "text": "✅ Allow Once", "payload": f"ea:once:{approval_id}"},
                        {"type": "callback", "text": "✅ Session", "payload": f"ea:session:{approval_id}"},
                    ],
                    [
                        {"type": "callback", "text": "✅ Always", "payload": f"ea:always:{approval_id}"},
                        {"type": "callback", "text": "❌ Deny", "payload": f"ea:deny:{approval_id}"},
                    ],
                ]
            },
        }
        try:
            msg = await self._require_client().send_message(
                chat_id=self._target_chat_id(chat_id),
                text=text,
                attachments=[keyboard],
                notify=True,
            )
            self._approval_state[approval_id] = session_key
            return SendResult(success=True, message_id=_as_str(_first_present(msg, "id", "message_id")), raw_response=msg)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def _handle_pymax_message(self, message: Any, client: Any = None) -> None:
        event = self._message_to_event(message)
        if event is None:
            return
        if self.mark_read and hasattr(message, "read"):
            try:
                result = message.read()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.debug("[MAX_USERBOT] mark_read failed for message %s: %s", event.message_id, exc)
        if self.download_media:
            await self._cache_pymax_media(event)
        await self.handle_message(event)

    async def _handle_pymax_raw(self, raw: Any, client: Any = None) -> None:
        payload = self._raw_callback_payload(raw)
        if payload:
            await self._handle_approval_callback(payload)

    def _message_to_event(self, message: Any) -> Optional[MessageEvent]:
        chat_id = _as_str(_first_present(message, "chat_id", "chatId", "dialog_id")).strip()
        user_id = _as_str(_first_present(message, "sender", "sender_id", "user_id", "from_id")).strip()
        if not chat_id:
            return None
        if self._is_own_user(user_id):
            return None
        if not self._is_authorized(user_id, chat_id):
            return None
        text = _as_str(_first_present(message, "text", "message", "body"))
        attaches = list(_first_present(message, "attaches", "attachments") or [])
        media_urls, media_types = self._attachment_media(message, attaches)
        if not text.strip() and not media_urls:
            return None
        source = self._source_for_message(message, chat_id, user_id)
        message_type = MessageType.TEXT
        if media_types and media_types[0].startswith("image/"):
            message_type = MessageType.PHOTO
        elif any(mt.startswith("audio/") or mt == "audio" for mt in media_types):
            message_type = MessageType.AUDIO
        elif any(mt.startswith("video/") or mt == "video" for mt in media_types):
            message_type = MessageType.DOCUMENT
        elif media_urls:
            message_type = MessageType.DOCUMENT
        return MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            raw_message=message,
            message_id=_as_str(_first_present(message, "id", "message_id")) or None,
            timestamp=self._timestamp(message),
            media_urls=media_urls,
            media_types=media_types,
        )

    def _source_for_message(self, message: Any, chat_id: str, user_id: str):
        from gateway.session import SessionSource

        chat_type = _as_str(_first_present(message, "chat_type", "type") or "dm").lower()
        if chat_type not in {"dm", "group", "channel", "forum"}:
            chat_type = "group" if chat_id != user_id else "dm"
        user_name = _as_str(_first_present(message, "sender_name", "user_name", "username", "author")) or None
        chat_name = _as_str(_first_present(message, "chat_name", "title")) or None
        return SessionSource(
            platform=Platform("max_userbot"),
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=user_id or None,
            user_name=user_name,
        )

    def _attachment_media(self, message: Any, attaches: list[Any]) -> tuple[list[str], list[str]]:
        urls: list[str] = []
        types: list[str] = []
        chat_id = _as_str(_first_present(message, "chat_id", "chatId", "dialog_id"))
        message_id = _as_str(_first_present(message, "id", "message_id"))
        for attach in attaches:
            kind = self._attachment_kind(attach)
            url = _as_str(_first_present(attach, "url", "base_url", "download_url", "href")).strip()
            mime = self._attachment_mime(attach, kind, url)
            if not url and kind == "file":
                file_id = _as_str(_first_present(attach, "file_id", "id")).strip()
                if file_id:
                    url = f"max_userbot_file:{chat_id}:{message_id}:{file_id}"
            if not url and kind == "video":
                video_id = _as_str(_first_present(attach, "video_id", "id")).strip()
                if video_id:
                    url = f"max_userbot_video:{chat_id}:{message_id}:{video_id}"
            if not url:
                continue
            urls.append(url)
            types.append(mime)
        return urls, types

    @staticmethod
    def _attachment_kind(attach: Any) -> str:
        raw = _as_str(_first_present(attach, "type", "_type", "attachment_type")).lower()
        cls_name = attach.__class__.__name__.lower()
        combined = f"{raw} {cls_name}"
        if "photo" in combined or "image" in combined:
            return "photo"
        if "audio" in combined or "voice" in combined:
            return "audio"
        if "video" in combined:
            return "video"
        if "file" in combined or "document" in combined:
            return "file"
        return raw or "file"

    @staticmethod
    def _attachment_mime(attach: Any, kind: str, url: str) -> str:
        explicit = _as_str(_first_present(attach, "mime_type", "mimeType", "content_type", "contentType")).lower()
        if explicit:
            return explicit
        if kind == "photo":
            return mimetypes.guess_type(url)[0] or "image/jpeg"
        if kind == "audio":
            return mimetypes.guess_type(url)[0] or "audio/mpeg"
        if kind == "video":
            return mimetypes.guess_type(url)[0] or "video/mp4"
        name = _as_str(_first_present(attach, "name", "filename", "file_name"))
        return mimetypes.guess_type(name or url)[0] or "application/octet-stream"

    async def _cache_pymax_media(self, event: MessageEvent) -> None:
        if not event.media_urls:
            return
        cached: list[str] = []
        for index, url in enumerate(event.media_urls):
            media_type = event.media_types[index] if index < len(event.media_types) else ""
            try:
                if url.startswith("max_userbot_file:"):
                    cached.append(await self._download_pymax_file_url(url))
                elif url.startswith("max_userbot_video:"):
                    cached.append(await self._download_pymax_video_url(url))
                elif media_type.startswith("image/"):
                    cached.append(await cache_image_from_url(url))
                else:
                    cached.append(url)
            except Exception as exc:
                logger.warning("[MAX_USERBOT] Failed to cache media %s: %s", safe_url_for_log(url), exc)
                cached.append(url)
        event.media_urls = cached

    async def _download_pymax_file_url(self, marker: str) -> str:
        _, chat_id, message_id, file_id = marker.split(":", 3)
        method = getattr(self._require_client(), "get_file_by_id", None)
        if method is None:
            return marker
        info = await method(chat_id=int(chat_id), message_id=int(message_id), file_id=int(file_id))
        url = _as_str(_first_present(info, "url"))
        # Hermes already has robust URL caching/downloading for Bot API; PyMax
        # can expose only a temporary URL here. Keep it as URL if no bytes API.
        return url or marker

    async def _download_pymax_video_url(self, marker: str) -> str:
        _, chat_id, message_id, video_id = marker.split(":", 3)
        method = getattr(self._require_client(), "get_video_by_id", None)
        if method is None:
            return marker
        info = await method(chat_id=int(chat_id), message_id=int(message_id), video_id=int(video_id))
        return _as_str(_first_present(info, "url")) or marker

    def _is_own_user(self, user_id: str) -> bool:
        return bool(user_id and self.account_id and user_id == self.account_id)

    def _is_authorized(self, user_id: str, chat_id: str) -> bool:
        if self.allow_all_users or _coerce_bool(os.getenv("GATEWAY_ALLOW_ALL_USERS"), False):
            return True
        allowed_users = set(self.allowed_users)
        allowed_users.update(_split_csv(os.getenv("GATEWAY_ALLOWED_USERS")))
        allowed_chats = set(self.allowed_chats)
        if user_id and (user_id in allowed_users or "*" in allowed_users):
            return True
        if chat_id and (chat_id in allowed_chats or "*" in allowed_chats):
            return True
        return False

    @staticmethod
    def _timestamp(message: Any) -> datetime:
        raw = _first_present(message, "time", "timestamp", "created_at")
        try:
            value = float(raw)
            if value > 10_000_000_000:
                value = value / 1000.0
            return datetime.fromtimestamp(value)
        except Exception:
            return datetime.now()

    @staticmethod
    def _target_chat_id(value: Any) -> Any:
        raw = str(value).strip()
        if raw.startswith("chat:") or raw.startswith("user:"):
            raw = raw.split(":", 1)[1]
        parsed = _coerce_int(raw)
        return parsed if parsed is not None else raw

    @staticmethod
    def _raw_callback_payload(raw: Any) -> Optional[dict[str, Any]]:
        data = raw
        if not isinstance(data, dict):
            data = getattr(raw, "data", None) or getattr(raw, "payload", None)
        if not isinstance(data, dict):
            return None
        callback_obj = data.get("callback")
        candidate = callback_obj if isinstance(callback_obj, dict) else data
        user_obj = candidate.get("user")
        user_container = user_obj if isinstance(user_obj, dict) else candidate
        payload = _as_str(_first_present(candidate, "payload", "data")).strip()
        callback_id = _as_str(_first_present(candidate, "callback_id", "id")).strip()
        user_id = _as_str(_first_present(user_container, "user_id", "sender", "sender_id", "id")).strip()
        if payload.startswith("ea:"):
            return {"payload": payload, "callback_id": callback_id, "user_id": user_id}
        return None

    async def _handle_approval_callback(self, callback: dict[str, Any]) -> None:
        payload = _as_str(callback.get("payload")).strip()
        parts = payload.split(":", 2)
        if len(parts) != 3:
            return
        choice = parts[1]
        if choice not in {"once", "session", "always", "deny"}:
            return
        try:
            approval_id = int(parts[2])
        except ValueError:
            return
        user_id = _as_str(callback.get("user_id")).strip()
        if not self._is_authorized(user_id, ""):
            return
        session_key = self._approval_state.pop(approval_id, None)
        if not session_key:
            return
        try:
            from tools.approval import resolve_gateway_approval
            resolve_gateway_approval(session_key, choice)
        except Exception as exc:
            logger.error("Failed to resolve MAX userbot approval callback: %s", exc)


MAX_USERBOT_PLATFORM_HINT = (
    "You are on MAX messenger through an unofficial internal MAX user account client, "
    "not the official MAX Bot API. Keep responses chat-friendly and concise. "
    "The account can send text, replies, edits, and native files/images through PyMax. "
    "MEDIA directives should be plain lines with real local paths; do not wrap MEDIA "
    "lines in inline code or fenced code blocks."
)


def register(ctx) -> None:
    ctx.register_platform(
        name="max_userbot",
        label="MAX Userbot",
        adapter_factory=lambda cfg: MaxUserbotAdapter(cfg),
        check_fn=check_max_userbot_requirements,
        validate_config=validate_max_userbot_config,
        is_connected=is_max_userbot_connected,
        required_env=["MAX_USERBOT_PHONE"],
        allowed_users_env="MAX_USERBOT_ALLOWED_USERS",
        allow_all_env="MAX_USERBOT_ALLOW_ALL_USERS",
        max_message_length=MAX_USERBOT_TEXT_LENGTH,
        platform_hint=MAX_USERBOT_PLATFORM_HINT,
        install_hint="Install PyMax with `uv pip install maxapi-python` and bootstrap a session before running as a service.",
        emoji="👤",
    )
