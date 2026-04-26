"""MAX messenger platform adapter.

Webhook-first, text-only MAX Bot API integration for Hermes gateway.
An explicit MAX_TRANSPORT=polling mode is available for local development/testing
without a public HTTPS URL; webhook remains the production default.

Official MAX Bot API semantics used here:
- production inbound delivery is Webhook via POST /subscriptions;
- MAX sends each webhook as an HTTPS POST containing an Update object;
- webhook secrets are verified through X-Max-Bot-Api-Secret;
- outbound text messages use POST /messages;
- explicit dev/test polling uses GET /updates;
- bot token is sent in the Authorization header;
- text payload uses NewMessageBody fields, with a 4000-character text limit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is installed in normal Hermes envs
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.platforms.helpers import MessageDeduplicator
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://platform-api.max.ru"
DEFAULT_WEBHOOK_HOST = "127.0.0.1"
DEFAULT_WEBHOOK_PORT = 8647
DEFAULT_WEBHOOK_PATH = "/max-webhook"
DEFAULT_UPDATE_TYPES = ("message_created", "bot_started")
DEFAULT_TRANSPORT = "webhook"
TRANSPORT_WEBHOOK = "webhook"
TRANSPORT_POLLING = "polling"
MAX_TEXT_LENGTH = 4000
_SECRET_RE = re.compile(r"^[a-zA-Z0-9_-]{5,256}$")


def _normalize_transport(value: Any) -> str:
    raw = str(value or DEFAULT_TRANSPORT).strip().lower().replace("-", "_")
    if raw in {"polling", "long_polling", "longpolling"}:
        return TRANSPORT_POLLING
    return TRANSPORT_WEBHOOK


def check_max_requirements() -> bool:
    """Return True when MAX is configured enough to start."""
    if not os.getenv("MAX_BOT_TOKEN", "").strip():
        return False
    if _normalize_transport(os.getenv("MAX_TRANSPORT")) == TRANSPORT_POLLING:
        return True
    return AIOHTTP_AVAILABLE


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _first_present(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


class MaxAdapter(BasePlatformAdapter):
    """Webhook-first text-only adapter for MAX messenger bots."""

    MAX_MESSAGE_LENGTH = MAX_TEXT_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MAX)
        extra = config.extra or {}
        self.token = (config.token or extra.get("token") or os.getenv("MAX_BOT_TOKEN", "")).strip()
        self.base_url = str(extra.get("base_url") or os.getenv("MAX_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.bot_user_id = _as_str(extra.get("bot_user_id") or os.getenv("MAX_BOT_USER_ID")).strip()
        self.bot_username = _as_str(extra.get("bot_username") or os.getenv("MAX_BOT_USERNAME")).strip().lstrip("@")
        self.webhook_host = str(extra.get("webhook_host") or os.getenv("MAX_WEBHOOK_HOST") or DEFAULT_WEBHOOK_HOST)
        self.webhook_port = int(extra.get("webhook_port") or os.getenv("MAX_WEBHOOK_PORT") or DEFAULT_WEBHOOK_PORT)
        self.webhook_path = str(extra.get("webhook_path") or os.getenv("MAX_WEBHOOK_PATH") or DEFAULT_WEBHOOK_PATH)
        if not self.webhook_path.startswith("/"):
            self.webhook_path = f"/{self.webhook_path}"
        self.webhook_public_url = str(
            extra.get("webhook_public_url") or os.getenv("MAX_WEBHOOK_PUBLIC_URL") or ""
        ).strip()
        self.webhook_secret = str(extra.get("webhook_secret") or os.getenv("MAX_WEBHOOK_SECRET") or "").strip()
        self.transport = _normalize_transport(extra.get("transport") or os.getenv("MAX_TRANSPORT") or DEFAULT_TRANSPORT)
        self.auto_subscribe = bool(
            extra.get("auto_subscribe")
            if "auto_subscribe" in extra
            else _env_bool("MAX_AUTO_SUBSCRIBE", True)
        )
        self.update_types = _split_csv(extra.get("update_types") or os.getenv("MAX_UPDATE_TYPES")) or list(DEFAULT_UPDATE_TYPES)
        self._client: Optional[httpx.AsyncClient] = None
        self._runner = None
        self._polling_task: Optional[asyncio.Task] = None
        self._poll_marker: Any = extra.get("polling_marker")
        self._dedup = MessageDeduplicator(max_size=5000, ttl_seconds=600)
        self._background_tasks: set[asyncio.Task] = set()

    @property
    def name(self) -> str:
        return "MAX"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": self.token,
            "Content-Type": "application/json",
        }

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        return self._client

    async def connect(self) -> bool:
        """Start the configured MAX inbound transport."""
        if not self.token:
            logger.error("[MAX] MAX_BOT_TOKEN is not configured")
            return False
        if self.transport == TRANSPORT_POLLING:
            return await self._connect_polling()
        if not AIOHTTP_AVAILABLE:
            logger.error("[MAX] aiohttp is required for webhook delivery")
            return False
        if self.webhook_secret and not _SECRET_RE.match(self.webhook_secret):
            logger.error(
                "[MAX] MAX_WEBHOOK_SECRET must match ^[a-zA-Z0-9_-]{5,256}$"
            )
            return False
        if self._running:
            return True

        app = web.Application()  # type: ignore[union-attr]
        app.router.add_get("/health", self._handle_health)
        app.router.add_post(self.webhook_path, self._handle_webhook)

        self._runner = web.AppRunner(app)  # type: ignore[union-attr]
        await self._runner.setup()
        try:
            site = web.TCPSite(self._runner, self.webhook_host, self.webhook_port)  # type: ignore[union-attr]
            await site.start()
        except Exception:
            await self._runner.cleanup()
            self._runner = None
            raise

        self._get_client()
        self._mark_connected()
        logger.info(
            "[MAX] webhook receiver listening on http://%s:%s%s",
            self.webhook_host,
            self.webhook_port,
            self.webhook_path,
        )

        if self.auto_subscribe:
            if self.webhook_public_url:
                await self._subscribe_webhook()
            else:
                logger.warning(
                    "[MAX] MAX_WEBHOOK_PUBLIC_URL is not configured; webhook receiver is running, "
                    "but MAX will not deliver events until /subscriptions is configured."
                )
        return True

    async def disconnect(self) -> None:
        """Stop webhook receiver and close HTTP resources."""
        self._mark_disconnected()
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._polling_task is not None:
            self._polling_task.cancel()
            await asyncio.gather(self._polling_task, return_exceptions=True)
            self._polling_task = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _connect_polling(self) -> bool:
        """Start dev/test long polling against MAX GET /updates."""
        if self._running:
            return True
        self._get_client()
        self._mark_connected()
        self._polling_task = asyncio.create_task(self._poll_loop(), name="max-polling")
        logger.warning(
            "[MAX] MAX_TRANSPORT=polling enabled; this is intended for local development/testing. "
            "Use webhook transport for production."
        )
        return True

    async def _poll_loop(self) -> None:
        backoff = 1.0
        while self._running:
            try:
                updates = await self._poll_once(timeout=30)
                if updates is None:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue
                backoff = 1.0
                for update in updates:
                    await self._handle_update(update)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[MAX] polling error: %s", exc, exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _poll_once(self, *, timeout: int = 30) -> Optional[list[Dict[str, Any]]]:
        params: Dict[str, Any] = {"timeout": timeout}
        if self._poll_marker is not None:
            params["marker"] = self._poll_marker
        response = await self._get_client().get(
            f"{self.base_url}/updates",
            params=params,
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return []
        marker = data.get("marker")
        if marker is not None:
            self._poll_marker = marker
        updates = data.get("updates")
        if not isinstance(updates, list):
            return []
        return [update for update in updates if isinstance(update, dict)]

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "platform": "max"})  # type: ignore[union-attr]

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        target = str(chat_id).strip()
        target_type = "user" if target.startswith("user:") else "chat"
        return {"id": target, "name": target, "type": target_type}

    def _validate_public_webhook_url(self) -> None:
        parsed = urlparse(self.webhook_public_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("MAX_WEBHOOK_PUBLIC_URL must be an HTTPS URL")
        if parsed.port is not None:
            raise ValueError("MAX webhook URL must not include a port; MAX supports HTTPS port 443 only")

    def _subscription_payload(self) -> Dict[str, Any]:
        self._validate_public_webhook_url()
        payload: Dict[str, Any] = {
            "url": self.webhook_public_url,
            "update_types": self.update_types,
        }
        if self.webhook_secret:
            payload["secret"] = self.webhook_secret
        return payload

    async def _subscribe_webhook(self) -> Dict[str, Any]:
        payload = self._subscription_payload()
        response = await self._get_client().post(
            f"{self.base_url}/subscriptions",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json()
        logger.info("[MAX] webhook subscription configured for %s", self.webhook_public_url)
        return data if isinstance(data, dict) else {"success": True}

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        if self.webhook_secret:
            supplied = request.headers.get("X-Max-Bot-Api-Secret", "")
            if supplied != self.webhook_secret:
                return web.json_response({"error": "unauthorized"}, status=401)  # type: ignore[union-attr]
        try:
            update = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)  # type: ignore[union-attr]
        if not isinstance(update, dict):
            return web.json_response({"error": "invalid_update"}, status=400)  # type: ignore[union-attr]

        task = asyncio.create_task(self._handle_update(update), name="max-webhook-update")
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return web.json_response({"success": True})  # type: ignore[union-attr]

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message through MAX POST /messages."""
        metadata = metadata or {}
        if not self.token:
            return SendResult(success=False, error="MAX_BOT_TOKEN is not configured")
        if not content:
            return SendResult(success=False, error="MAX text message is empty")

        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH)
        last_result: Optional[SendResult] = None
        for chunk in chunks:
            params = self._target_params(chat_id, metadata)
            body: Dict[str, Any] = {"text": chunk}
            if reply_to:
                body["link"] = {"type": "reply", "mid": reply_to}
            if "notify" in metadata:
                body["notify"] = bool(metadata["notify"])
            if metadata.get("format"):
                body["format"] = metadata["format"]
            try:
                response = await self._get_client().post(
                    f"{self.base_url}/messages",
                    params=params,
                    json=body,
                    headers=self._headers(),
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                return SendResult(success=False, error=str(exc), retryable=True)

            message_id = self._extract_message_id(payload)
            last_result = SendResult(success=True, message_id=message_id, raw_response=payload)
        return last_result or SendResult(success=False, error="No MAX message chunks were sent")

    @staticmethod
    def _target_params(target: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        target_type = str(metadata.get("target_type") or "chat").lower().strip()
        raw = str(target).strip()
        if raw.startswith("user:"):
            return {"user_id": raw.split(":", 1)[1]}
        if raw.startswith("chat:"):
            return {"chat_id": raw.split(":", 1)[1]}
        if target_type == "user":
            return {"user_id": raw}
        return {"chat_id": raw}

    @staticmethod
    def _extract_message_id(payload: Any) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        value = _first_present(payload, "message_id", "id", "mid")
        if value is None:
            value = _first_present(message, "message_id", "id", "mid")
        return _as_str(value) if value is not None else None

    async def _handle_update(self, update: Dict[str, Any]) -> None:
        event = self._update_to_event(update)
        if event is None:
            return
        dedup_key = self._dedup_key(update, event)
        if self._dedup.is_duplicate(dedup_key):
            return
        await self.handle_message(event)

    def _update_to_event(self, update: Dict[str, Any]) -> Optional[MessageEvent]:
        if update.get("update_type") != "message_created":
            return None
        message = update.get("message")
        if not isinstance(message, dict):
            return None
        sender = message.get("sender") if isinstance(message.get("sender"), dict) else {}
        if self._is_own_message(sender):
            return None
        body = message.get("body") if isinstance(message.get("body"), dict) else {}
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        recipient = message.get("recipient") if isinstance(message.get("recipient"), dict) else {}
        chat_id = self._chat_id(sender, recipient)
        if not chat_id:
            return None
        source = SessionSource(
            platform=Platform.MAX,
            chat_id=chat_id,
            chat_name=self._chat_name(recipient),
            chat_type=self._chat_type(recipient),
            user_id=_as_str(sender.get("user_id") or sender.get("id")).strip() or None,
            user_name=self._user_name(sender),
            is_bot=bool(sender.get("is_bot")),
        )
        message_id = self._extract_message_id(message)
        timestamp = self._timestamp(update, message)
        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=update,
            message_id=message_id,
            timestamp=timestamp,
        )

    def _is_own_message(self, sender: Dict[str, Any]) -> bool:
        sender_id = _as_str(sender.get("user_id") or sender.get("id")).strip()
        if self.bot_user_id and sender_id == self.bot_user_id:
            return True
        username = _as_str(sender.get("username")).strip().lstrip("@")
        return bool(self.bot_username and username and username.lower() == self.bot_username.lower())

    @staticmethod
    def _chat_id(sender: Dict[str, Any], recipient: Dict[str, Any]) -> str:
        chat_id = _first_present(recipient, "chat_id", "id")
        if chat_id is not None:
            return _as_str(chat_id)
        # Dialog updates may identify the peer only through the sender object.
        return _as_str(_first_present(sender, "user_id", "id"))

    @staticmethod
    def _chat_name(recipient: Dict[str, Any]) -> Optional[str]:
        value = _first_present(recipient, "title", "name")
        return _as_str(value) if value is not None else None

    @staticmethod
    def _chat_type(recipient: Dict[str, Any]) -> str:
        raw = _as_str(recipient.get("type")).lower()
        if raw == "channel":
            return "channel"
        if raw == "chat":
            return "group"
        return "dm"

    @staticmethod
    def _user_name(sender: Dict[str, Any]) -> Optional[str]:
        first = _as_str(sender.get("first_name")).strip()
        last = _as_str(sender.get("last_name")).strip()
        full = " ".join(part for part in (first, last) if part).strip()
        if full:
            return full
        username = _as_str(sender.get("username")).strip()
        return username or None

    @staticmethod
    def _timestamp(update: Dict[str, Any], message: Dict[str, Any]) -> datetime:
        raw = _first_present(message, "timestamp", "created_at")
        if raw is None:
            raw = update.get("timestamp")
        try:
            value = float(raw)
            # MAX docs use Unix-time. Be tolerant of millisecond timestamps.
            if value > 10_000_000_000:
                value = value / 1000.0
            return datetime.fromtimestamp(value)
        except Exception:
            return datetime.now()

    @staticmethod
    def _dedup_key(update: Dict[str, Any], event: MessageEvent) -> str:
        explicit = _first_present(update, "update_id", "id", "marker")
        if explicit is not None:
            return f"max:update:{explicit}"
        return "max:message:{chat}:{user}:{ts}:{text}".format(
            chat=event.source.chat_id,
            user=event.source.user_id or "",
            ts=int(event.timestamp.timestamp()),
            text=event.text[:128],
        )
