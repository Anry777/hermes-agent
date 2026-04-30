"""Provider proxy mode for the OpenAI-compatible API server.

This module implements the raw/compat proxy path used when the API server is
configured with ``extra.mode: provider_proxy``.  It deliberately bypasses
AIAgent: no Hermes prompts, skills, tools, memory, or session state are applied.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - imported only when aiohttp is present
    web = None  # type: ignore[assignment]

from hermes_cli.runtime_provider import resolve_runtime_provider
from run_agent import OpenAI

logger = logging.getLogger(__name__)

_MAX_MODEL_ID_LEN = 256
_INVALID_MODEL_ID_CHARS = re.compile(r"[\r\n\x00]")


@dataclass(frozen=True)
class ProxyModelSpec:
    """A public model id mapped to an internal Hermes provider/model target."""

    public_id: str
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_mode: Optional[str] = None
    owned_by: Optional[str] = None
    request_defaults: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> Optional["ProxyModelSpec"]:
        public_id = str(raw.get("id") or raw.get("public_id") or "").strip()
        provider = str(raw.get("provider") or "").strip()
        model = str(raw.get("model") or "").strip()
        if not public_id or not provider or not model:
            return None
        if len(public_id) > _MAX_MODEL_ID_LEN or _INVALID_MODEL_ID_CHARS.search(public_id):
            return None
        request_defaults = raw.get("request_defaults")
        if not isinstance(request_defaults, dict):
            request_defaults = {}
        return cls(
            public_id=public_id,
            provider=provider,
            model=model,
            base_url=str(raw.get("base_url") or "").strip() or None,
            api_key=str(raw.get("api_key") or "").strip() or None,
            api_mode=str(raw.get("api_mode") or "").strip() or None,
            owned_by=str(raw.get("owned_by") or provider).strip() or provider,
            request_defaults=dict(request_defaults),
        )


@dataclass(frozen=True)
class ResolvedProxyTarget:
    public_id: str
    provider: str
    model: str
    api_mode: str
    base_url: str
    api_key: str
    runtime: Dict[str, Any]


def _openai_error(
    message: str,
    *,
    err_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    error: Dict[str, Any] = {"message": message, "type": err_type}
    if param is not None:
        error["param"] = param
    if code is not None:
        error["code"] = code
    return {"error": error}


def _object_to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    for method in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, method, None)
        if callable(fn):
            data = fn()
            if isinstance(data, dict):
                return data
    try:
        return json.loads(obj.model_dump_json())
    except Exception:
        return {}


def _usage_to_openai_dict(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or prompt + completion)
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
    prompt = int(getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0)
    completion = int(getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0)
    total = int(getattr(usage, "total_tokens", None) or prompt + completion)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _chat_completion_from_normalized(public_model: str, normalized: Any, raw_response: Any = None) -> Dict[str, Any]:
    content = getattr(normalized, "content", None) or ""
    finish_reason = getattr(normalized, "finish_reason", None) or "stop"
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    tool_calls = getattr(normalized, "tool_calls", None)
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id or f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in tool_calls
        ]
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": public_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage_to_openai_dict(raw_response),
    }


class APIServerProviderProxy:
    """OpenAI-compatible raw/compat provider proxy for API Server."""

    def __init__(self, config: Dict[str, Any]):
        self._default_model = str(config.get("default_model") or "").strip()
        self._require_explicit_model = bool(config.get("require_explicit_model", True))
        self._allow_streaming = bool(config.get("allow_streaming", False))
        self._models = self._parse_models(config.get("models") or [])
        self._models_by_id = {model.public_id: model for model in self._models}

    @classmethod
    def from_extra(cls, extra: Dict[str, Any]) -> Optional["APIServerProviderProxy"]:
        mode = str(extra.get("mode") or "").strip().lower()
        proxy_config = extra.get("provider_proxy")
        if not isinstance(proxy_config, dict):
            proxy_config = {}
        enabled = bool(proxy_config.get("enabled", False)) or mode == "provider_proxy"
        if not enabled:
            return None
        return cls(proxy_config)

    @property
    def enabled(self) -> bool:
        return True

    @staticmethod
    def _parse_models(raw_models: Any) -> List[ProxyModelSpec]:
        if not isinstance(raw_models, list):
            return []
        models: List[ProxyModelSpec] = []
        seen: set[str] = set()
        for raw in raw_models:
            if not isinstance(raw, dict):
                continue
            spec = ProxyModelSpec.from_dict(raw)
            if spec is None or spec.public_id in seen:
                continue
            seen.add(spec.public_id)
            models.append(spec)
        return models

    def models_response(self) -> Dict[str, Any]:
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": spec.public_id,
                    "object": "model",
                    "created": created,
                    "owned_by": spec.owned_by or spec.provider,
                    "permission": [],
                    "root": spec.public_id,
                    "parent": None,
                }
                for spec in self._models
            ],
        }

    async def handle_models(self, request: "web.Request") -> "web.Response":
        return web.json_response(self.models_response())

    async def handle_chat_completions(self, request: "web.Request") -> "web.Response":
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        if body.get("stream") and not self._allow_streaming:
            return web.json_response(
                _openai_error(
                    "Streaming is not supported in provider_proxy mode yet",
                    code="unsupported_streaming",
                ),
                status=400,
            )

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return web.json_response(
                _openai_error("Missing or invalid 'messages' field", param="messages"),
                status=400,
            )

        model_id = str(body.get("model") or "").strip()
        if not model_id and not self._require_explicit_model:
            model_id = self._default_model
        if not model_id:
            return web.json_response(_openai_error("Missing required 'model' field", param="model"), status=400)
        if len(model_id) > _MAX_MODEL_ID_LEN or _INVALID_MODEL_ID_CHARS.search(model_id):
            return web.json_response(_openai_error("Invalid model id", param="model"), status=400)

        spec = self._models_by_id.get(model_id)
        if spec is None:
            return web.json_response(
                _openai_error(
                    f"The model '{model_id}' does not exist or is not enabled for this endpoint.",
                    param="model",
                    code="model_not_found",
                ),
                status=404,
            )

        try:
            target = self._resolve_target(spec)
            if target.api_mode == "chat_completions":
                data = await self._call_chat_completions(body, target, spec)
            elif target.api_mode == "codex_responses":
                data = await self._call_codex_chat_compat(body, target, spec)
            else:
                return web.json_response(
                    _openai_error(
                        f"Provider proxy does not support api_mode '{target.api_mode}' for chat completions yet.",
                        code="unsupported_operation",
                    ),
                    status=400,
                )
        except Exception as exc:
            logger.exception("Provider proxy chat completion failed for model %s", spec.public_id)
            return web.json_response(
                _openai_error(f"Provider proxy request failed: {exc}", err_type="server_error", code="provider_proxy_error"),
                status=502,
            )

        return web.json_response(data)

    def _resolve_target(self, spec: ProxyModelSpec) -> ResolvedProxyTarget:
        runtime = resolve_runtime_provider(
            requested=spec.provider,
            explicit_base_url=spec.base_url,
            explicit_api_key=spec.api_key,
            target_model=spec.model,
        )
        api_mode = spec.api_mode or str(runtime.get("api_mode") or "chat_completions")
        return ResolvedProxyTarget(
            public_id=spec.public_id,
            provider=str(runtime.get("provider") or spec.provider),
            model=spec.model,
            api_mode=api_mode,
            base_url=str(runtime.get("base_url") or ""),
            api_key=str(runtime.get("api_key") or ""),
            runtime=runtime,
        )

    @staticmethod
    def _create_openai_client(target: ResolvedProxyTarget) -> Any:
        kwargs: Dict[str, Any] = {}
        if target.api_key:
            kwargs["api_key"] = target.api_key
        if target.base_url:
            kwargs["base_url"] = target.base_url
        return OpenAI(**kwargs)

    async def _call_chat_completions(
        self,
        body: Dict[str, Any],
        target: ResolvedProxyTarget,
        spec: ProxyModelSpec,
    ) -> Dict[str, Any]:
        payload = dict(spec.request_defaults)
        payload.update(body)
        payload["model"] = target.model
        payload.pop("stream", None)

        def _call() -> Dict[str, Any]:
            client = self._create_openai_client(target)
            try:
                response = client.chat.completions.create(**payload)
                data = _object_to_dict(response)
                if not data:
                    data = _chat_completion_from_normalized(spec.public_id, None, response)
                data["model"] = spec.public_id
                return data
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

        return await asyncio.get_running_loop().run_in_executor(None, _call)

    async def _call_codex_chat_compat(
        self,
        body: Dict[str, Any],
        target: ResolvedProxyTarget,
        spec: ProxyModelSpec,
    ) -> Dict[str, Any]:
        from agent.codex_responses_adapter import _chat_messages_to_responses_input, _responses_tools
        from agent.transports.codex import ResponsesApiTransport

        messages = body.get("messages") or []
        instructions = "\n".join(
            str(msg.get("content") or "")
            for msg in messages
            if isinstance(msg, dict) and msg.get("role") in {"system", "developer"}
        ).strip()
        if not instructions:
            # Codex Responses requires a non-empty instructions field even for
            # plain Chat Completions requests that contain only user messages.
            # Keep the fallback neutral so provider_proxy remains a raw-ish
            # compatibility bridge instead of injecting Hermes agent identity.
            instructions = str(
                spec.request_defaults.get("instructions")
                or body.get("instructions")
                or "You are a helpful assistant."
            ).strip()
        payload_messages = [
            msg for msg in messages
            if not (isinstance(msg, dict) and msg.get("role") in {"system", "developer"})
        ]
        payload: Dict[str, Any] = {
            "model": target.model,
            "input": _chat_messages_to_responses_input(payload_messages),
            "store": False,
        }
        if instructions:
            payload["instructions"] = instructions
        tools = _responses_tools(body.get("tools"))
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = body.get("tool_choice", "auto")
        # ChatGPT's Codex backend rejects max_output_tokens on its streaming
        # compatibility endpoint, while standard Responses-compatible backends
        # and tests expect the Chat Completions token limit to be mapped.
        is_chatgpt_codex_backend = "chatgpt.com/backend-api/codex" in target.base_url
        if not is_chatgpt_codex_backend and body.get("max_tokens") is not None:
            payload["max_output_tokens"] = body.get("max_tokens")
        if not is_chatgpt_codex_backend and body.get("max_completion_tokens") is not None:
            payload["max_output_tokens"] = body.get("max_completion_tokens")

        def _call() -> Dict[str, Any]:
            client = self._create_openai_client(target)
            try:
                # The ChatGPT Codex backend requires streaming Responses API
                # requests, but provider_proxy exposes a non-streaming
                # Chat Completions response to callers.  Stream internally,
                # collect the terminal response, then normalize it.
                stream_payload = dict(payload)
                stream_payload["stream"] = True
                stream_or_response = client.responses.create(**stream_payload)
                if hasattr(stream_or_response, "output") or not hasattr(stream_or_response, "__iter__"):
                    response = stream_or_response
                else:
                    response = self._collect_streaming_response(stream_or_response)
                normalized = ResponsesApiTransport().normalize_response(response)
                return _chat_completion_from_normalized(spec.public_id, normalized, response)
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

        return await asyncio.get_running_loop().run_in_executor(None, _call)

    @staticmethod
    def _collect_streaming_response(stream_or_response: Any) -> Any:
        terminal_response = None
        collected_output_items: List[Any] = []
        collected_text_deltas: List[str] = []
        try:
            for event in stream_or_response:
                event_type = getattr(event, "type", None)
                if not event_type and isinstance(event, dict):
                    event_type = event.get("type")
                if event_type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if item is None and isinstance(event, dict):
                        item = event.get("item")
                    if item is not None:
                        collected_output_items.append(item)
                elif event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if not delta and isinstance(event, dict):
                        delta = event.get("delta", "")
                    if delta:
                        collected_text_deltas.append(str(delta))
                elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
                    terminal_response = getattr(event, "response", None)
                    if terminal_response is None and isinstance(event, dict):
                        terminal_response = event.get("response")
                    if terminal_response is not None:
                        break
        finally:
            close_fn = getattr(stream_or_response, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        if terminal_response is None:
            raise RuntimeError("Responses stream did not emit a terminal response")
        output = getattr(terminal_response, "output", None)
        if isinstance(output, list) and not output:
            if collected_output_items:
                terminal_response.output = list(collected_output_items)
            elif collected_text_deltas:
                terminal_response.output = [SimpleNamespace(
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[SimpleNamespace(type="output_text", text="".join(collected_text_deltas))],
                )]
        return terminal_response
