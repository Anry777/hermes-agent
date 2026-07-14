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

import httpx

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
    credential_pool: Any = None
    credential_id: Optional[str] = None


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


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of SDK objects into JSON-serializable values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(item) for item in obj]
    for method in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                data = fn()
            except Exception:
                data = None
            if data is not None:
                return _to_jsonable(data)
    fn = getattr(obj, "model_dump_json", None)
    if callable(fn):
        try:
            return _to_jsonable(json.loads(fn()))
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {
            str(key): _to_jsonable(value)
            for key, value in vars(obj).items()
            if not str(key).startswith("_")
        }
    return str(obj)


def _object_to_dict(obj: Any) -> Dict[str, Any]:
    data = _to_jsonable(obj)
    if isinstance(data, dict):
        return data
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
            _openai_tool_call_from_normalized(tc)
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


class _ProxyContentError(ValueError):
    def __init__(self, message: str, *, param: str, code: str = "invalid_content_part"):
        super().__init__(message)
        self.param = param
        self.code = code


_CONTENT_TEXT_TYPES = {"text", "input_text", "output_text"}
_CONTENT_IMAGE_TYPES = {"image_url", "input_image"}
_CONTENT_UNSUPPORTED_FILE_TYPES = {"file", "input_file", "file_id"}
_CODEX_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}
_CODEX_REASONING_EFFORT_ALIASES = {"minimal": "low"}
_CODEX_CHATGPT_UNSUPPORTED_SAMPLING_PARAMS = {
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "logprobs",
    "top_logprobs",
}
_CODEX_RESPONSES_UNSUPPORTED_SDK_PARAMS = {
    # VSCode/Codex-style clients may send this as wire metadata, but the
    # OpenAI Python SDK does not accept it as a responses.create() kwarg.
    "client_metadata",
}


def _tool_call_attr(tool_call: Any, name: str, default: Any = None) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get(name, default)
    return getattr(tool_call, name, default)


def _tool_call_function_attr(tool_call: Any, name: str, default: Any = None) -> Any:
    fn = _tool_call_attr(tool_call, "function")
    if isinstance(fn, dict):
        return fn.get(name, default)
    if fn is not None:
        return getattr(fn, name, default)
    return _tool_call_attr(tool_call, name, default)


def _openai_tool_call_from_normalized(tool_call: Any, *, index: Optional[int] = None) -> Dict[str, Any]:
    call_id = _tool_call_attr(tool_call, "id") or _tool_call_attr(tool_call, "call_id")
    name = _tool_call_function_attr(tool_call, "name", "") or ""
    arguments = _tool_call_function_attr(tool_call, "arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    item: Dict[str, Any] = {
        "id": str(call_id or f"call_{uuid.uuid4().hex[:16]}"),
        "type": "function",
        "function": {"name": str(name), "arguments": arguments},
    }
    if index is not None:
        item["index"] = index
    return item


def _responses_tool_choice_from_chat(tool_choice: Any) -> Any:
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type") or "").strip()
        fn = tool_choice.get("function")
        if choice_type == "function" and isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str) and name.strip():
                return {"type": "function", "name": name.strip()}
    return tool_choice


def _normalize_codex_reasoning_effort(value: Any) -> Optional[str]:
    if value is None:
        return None
    effort = str(value).strip().lower()
    if not effort:
        return None
    effort = _CODEX_REASONING_EFFORT_ALIASES.get(effort, effort)
    if effort in _CODEX_REASONING_EFFORTS:
        return effort
    logger.debug("Dropping unsupported Codex reasoning effort %r", value)
    return None


def _responses_reasoning_from_chat(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        mapped = dict(reasoning)
        if "effort" in mapped:
            effort = _normalize_codex_reasoning_effort(mapped.get("effort"))
            if effort:
                mapped["effort"] = effort
            else:
                mapped.pop("effort", None)
        if mapped:
            return mapped
    elif isinstance(reasoning, str):
        effort = _normalize_codex_reasoning_effort(reasoning)
        if effort:
            return {"effort": effort}

    effort = _normalize_codex_reasoning_effort(body.get("reasoning_effort"))
    if effort:
        return {"effort": effort}

    effort = _normalize_codex_reasoning_effort(body.get("model_reasoning_effort"))
    if effort:
        return {"effort": effort}
    return None


def _is_chatgpt_codex_backend(target: ResolvedProxyTarget) -> bool:
    return "chatgpt.com/backend-api/codex" in target.base_url


def _should_pass_codex_request_param(key: str, *, is_chatgpt_codex_backend: bool) -> bool:
    if is_chatgpt_codex_backend and key in _CODEX_CHATGPT_UNSUPPORTED_SAMPLING_PARAMS:
        return False
    return True


def _validate_chat_messages_content(messages: List[Dict[str, Any]]) -> None:
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part_idx, part in enumerate(content):
            param = f"messages[{idx}].content"
            if isinstance(part, str):
                continue
            if not isinstance(part, dict):
                raise _ProxyContentError(
                    f"Content part {part_idx} must be an object or string.",
                    param=param,
                )
            ptype = str(part.get("type") or "").strip().lower()
            if ptype in _CONTENT_TEXT_TYPES:
                text = part.get("text", "")
                if text is not None and not isinstance(text, str):
                    raise _ProxyContentError(
                        f"Text content part {part_idx} must contain a string text field.",
                        param=param,
                    )
                continue
            if ptype in _CONTENT_IMAGE_TYPES:
                image_ref = part.get("image_url")
                if isinstance(image_ref, dict):
                    url = image_ref.get("url")
                else:
                    url = image_ref
                if not isinstance(url, str) or not url.strip():
                    raise _ProxyContentError(
                        "Image parts must include a non-empty image URL.",
                        param=param,
                        code="invalid_image_url",
                    )
                url = url.strip()
                if not (url.startswith("http://") or url.startswith("https://") or url.startswith("data:image/")):
                    raise _ProxyContentError(
                        "Image inputs must use http(s) URLs or data:image/... URLs.",
                        param=param,
                        code="invalid_image_url",
                    )
                continue
            if ptype in _CONTENT_UNSUPPORTED_FILE_TYPES or "file" in ptype:
                raise _ProxyContentError(
                    "File content parts are not supported by provider_proxy chat completions. Use text or image_url/input_image parts.",
                    param=param,
                    code="unsupported_content_type",
                )
            raise _ProxyContentError(
                f"Unsupported content part type {ptype!r}. Only text and image_url/input_image parts are supported.",
                param=param,
                code="unsupported_content_type",
            )


class APIServerProviderProxy:
    """OpenAI-compatible raw/compat provider proxy for API Server."""

    def __init__(self, config: Dict[str, Any], *, mode: str = "provider_proxy"):
        self._mode = mode
        self._default_model = str(config.get("default_model") or "").strip()
        self._require_explicit_model = bool(config.get("require_explicit_model", True))
        self._allow_streaming = bool(config.get("allow_streaming", False))
        self._models = self._parse_models(config.get("models") or [])
        self._models_by_id = {model.public_id: model for model in self._models}
        self._codex_provider = str(config.get("provider") or config.get("codex_provider") or "openai-codex").strip() or "openai-codex"
        self._model_discovery = str(config.get("model_discovery") or "catalog").strip().lower() or "catalog"
        self._allow_model_patterns = self._compile_patterns(config.get("allow_models") or config.get("allow_model_patterns"))
        self._deny_model_patterns = self._compile_patterns(config.get("deny_models") or config.get("deny_model_patterns"))
        self._include_done_event = bool(config.get("include_done_event", True))

    @classmethod
    def from_extra(cls, extra: Dict[str, Any]) -> Optional["APIServerProviderProxy"]:
        mode = str(extra.get("mode") or "").strip().lower()
        if mode == "codex_responses_proxy":
            codex_config = extra.get("codex_responses_proxy")
            if not isinstance(codex_config, dict):
                codex_config = {}
            return cls(codex_config, mode="codex_responses_proxy")

        codex_config = extra.get("codex_responses_proxy")
        if isinstance(codex_config, dict) and bool(codex_config.get("enabled", False)):
            return cls(codex_config, mode="codex_responses_proxy")

        proxy_config = extra.get("provider_proxy")
        if not isinstance(proxy_config, dict):
            proxy_config = {}
        enabled = bool(proxy_config.get("enabled", False)) or mode == "provider_proxy"
        if not enabled:
            return None
        return cls(proxy_config, mode="provider_proxy")

    @property
    def enabled(self) -> bool:
        return True

    @property
    def mode(self) -> str:
        return self._mode

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

    @staticmethod
    def _compile_patterns(raw_patterns: Any) -> List[re.Pattern[str]]:
        if not raw_patterns:
            return []
        if isinstance(raw_patterns, str):
            items = [raw_patterns]
        elif isinstance(raw_patterns, (list, tuple, set)):
            items = list(raw_patterns)
        else:
            items = [str(raw_patterns)]
        patterns: List[re.Pattern[str]] = []
        for item in items:
            pattern = str(item or "").strip()
            if not pattern:
                continue
            try:
                patterns.append(re.compile(pattern))
            except re.error:
                logger.warning("Ignoring invalid provider_proxy model filter pattern: %r", pattern)
        return patterns

    def _model_allowed(self, model_id: str) -> bool:
        if not model_id or len(model_id) > _MAX_MODEL_ID_LEN or _INVALID_MODEL_ID_CHARS.search(model_id):
            return False
        if self._allow_model_patterns and not any(pattern.search(model_id) for pattern in self._allow_model_patterns):
            return False
        if self._deny_model_patterns and any(pattern.search(model_id) for pattern in self._deny_model_patterns):
            return False
        return True

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
        if self._mode == "codex_responses_proxy":
            return await self._handle_codex_models(request)
        return web.json_response(self.models_response())

    async def _handle_codex_models(self, request: "web.Request") -> "web.Response":
        if self._model_discovery != "live":
            return web.json_response(self.models_response())

        seed_model = self._default_model
        if not seed_model and self._models:
            seed_model = self._models[0].model
        seed_model = seed_model or "gpt-5.4"

        try:
            target = self._resolve_codex_target(seed_model)

            def _list_models() -> Dict[str, Any]:
                if _is_chatgpt_codex_backend(target):
                    base_url = target.base_url.rstrip("/") or "https://chatgpt.com/backend-api/codex"
                    resp = httpx.get(
                        f"{base_url}/models",
                        params={"client_version": "1.0.0"},
                        headers={"Authorization": f"Bearer {target.api_key}"},
                        timeout=15,
                    )
                    resp.raise_for_status()
                    return resp.json()

                client = self._create_openai_client(target)
                try:
                    return _object_to_dict(
                        client.models.list(extra_query={"client_version": "1.0.0"})
                    )
                finally:
                    close = getattr(client, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass

            data = await asyncio.get_running_loop().run_in_executor(None, _list_models)
            raw_items = data.get("data") if isinstance(data, dict) else None
            raw_codex_items = data.get("models") if isinstance(data, dict) else None
            if not isinstance(raw_items, list) and isinstance(raw_codex_items, list):
                raw_items = raw_codex_items
            if not isinstance(raw_items, list):
                raw_items = []
            created = int(time.time())
            items: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for raw in raw_items:
                raw_dict = raw if isinstance(raw, dict) else _object_to_dict(raw)
                model_id = str(raw_dict.get("id") or raw_dict.get("slug") or "").strip()
                if not self._model_allowed(model_id) or model_id in seen:
                    continue
                seen.add(model_id)
                item = dict(raw_dict)
                item.setdefault("object", "model")
                item.setdefault("created", created)
                item.setdefault("owned_by", self._codex_provider)
                item.setdefault("permission", [])
                item.setdefault("root", model_id)
                item.setdefault("parent", None)
                item["id"] = model_id
                items.append(item)
            return web.json_response({"object": "list", "data": items})
        except Exception as exc:
            logger.exception("Codex Responses proxy model discovery failed")
            return web.json_response(
                _openai_error(
                    f"Codex Responses proxy model discovery failed: {exc}",
                    err_type="server_error",
                    code="codex_responses_proxy_error",
                ),
                status=502,
            )

    async def handle_chat_completions(self, request: "web.Request") -> "web.Response":
        if self._mode == "codex_responses_proxy":
            return web.json_response(
                _openai_error(
                    "Chat Completions is not supported in codex_responses_proxy mode; use /v1/responses",
                    code="unsupported_operation",
                ),
                status=400,
            )
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return web.json_response(
                _openai_error("Missing or invalid 'messages' field", param="messages"),
                status=400,
            )
        try:
            _validate_chat_messages_content(messages)
        except _ProxyContentError as exc:
            return web.json_response(
                _openai_error(str(exc), param=exc.param, code=exc.code),
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
            if body.get("stream"):
                if not self._allow_streaming:
                    return web.json_response(
                        _openai_error(
                            "Streaming is not enabled for provider_proxy mode",
                            code="unsupported_streaming",
                        ),
                        status=400,
                    )
                if target.api_mode == "chat_completions":
                    return await self._stream_chat_completions(request, body, target, spec)
                if target.api_mode == "codex_responses":
                    return await self._stream_codex_chat_compat(request, body, target, spec)
                return web.json_response(
                    _openai_error(
                        f"Provider proxy does not support streaming for api_mode '{target.api_mode}' yet.",
                        code="unsupported_operation",
                    ),
                    status=400,
                )
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

    async def handle_responses(self, request: "web.Request") -> "web.Response":
        if self._mode != "codex_responses_proxy":
            return web.json_response(
                _openai_error(
                    "The Responses API is not supported in provider_proxy mode yet",
                    code="unsupported_operation",
                ),
                status=400,
            )

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)
        if not isinstance(body, dict):
            return web.json_response(_openai_error("Request body must be a JSON object"), status=400)

        model_id = str(body.get("model") or self._default_model or "").strip()
        if not model_id:
            return web.json_response(_openai_error("Missing required 'model' field", param="model"), status=400)
        if not self._model_allowed(model_id):
            return web.json_response(_openai_error("Invalid model id", param="model"), status=400)
        if self._model_discovery != "live" and self._models_by_id and model_id not in self._models_by_id:
            return web.json_response(
                _openai_error(
                    f"The model '{model_id}' does not exist or is not enabled for this endpoint.",
                    param="model",
                    code="model_not_found",
                ),
                status=404,
            )

        try:
            target = self._resolve_codex_target(model_id)
            if target.api_mode != "codex_responses":
                return web.json_response(
                    _openai_error(
                        f"codex_responses_proxy requires api_mode 'codex_responses', got '{target.api_mode}'.",
                        code="unsupported_operation",
                    ),
                    status=400,
                )
            if body.get("stream"):
                return await self._stream_codex_responses_passthrough(request, body, target)
            data = await self._call_codex_responses_passthrough(body, target)
        except Exception as exc:
            logger.exception("Codex Responses proxy request failed for model %s", model_id)
            return web.json_response(
                _openai_error(
                    f"Codex Responses proxy request failed: {exc}",
                    err_type="server_error",
                    code="codex_responses_proxy_error",
                ),
                status=502,
            )
        return web.json_response(data)

    @staticmethod
    def _runtime_pool_context(runtime: Dict[str, Any]) -> tuple[Any, Optional[str]]:
        pool = runtime.get("credential_pool") if isinstance(runtime, dict) else None
        credential_id = None
        if pool is not None:
            current_fn = getattr(pool, "current", None)
            if callable(current_fn):
                try:
                    current = current_fn()
                except Exception:
                    current = None
                credential_id = getattr(current, "id", None) if current is not None else None
        return pool, str(credential_id) if credential_id else None

    @staticmethod
    def _exception_status_code(exc: Exception) -> Optional[int]:
        for attr in ("status_code", "status"):
            value = getattr(exc, attr, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    @staticmethod
    def _exception_error_context(exc: Exception) -> Dict[str, Any]:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                result = {
                    key: error.get(key)
                    for key in ("message", "code", "reason")
                    if error.get(key) is not None
                }
                if result:
                    return result
            detail = body.get("detail")
            if isinstance(detail, str) and detail.strip():
                return {"message": detail.strip()}
        message = str(exc).strip()
        return {"message": message} if message else {}

    def _record_target_success(self, target: ResolvedProxyTarget) -> None:
        pool = target.credential_pool
        if pool is None:
            return
        record_success = getattr(pool, "record_success", None)
        if not callable(record_success):
            return
        try:
            record_success(target.credential_id)
        except Exception as exc:
            logger.debug("Codex Responses proxy failed to record credential success: %s", exc)

    def _mark_target_failure(self, target: ResolvedProxyTarget, exc: Exception) -> None:
        status_code = self._exception_status_code(exc)
        if status_code not in {401, 402, 429}:
            return
        pool = target.credential_pool
        if pool is None:
            return
        mark = getattr(pool, "mark_exhausted_and_rotate", None)
        if not callable(mark):
            return
        try:
            mark(
                status_code=status_code,
                error_context=self._exception_error_context(exc),
                credential_id=target.credential_id,
                api_key_hint=target.api_key,
            )
        except Exception as mark_exc:
            logger.debug("Codex Responses proxy failed to mark credential failure: %s", mark_exc)

    def _resolve_codex_target(self, model_id: str) -> ResolvedProxyTarget:
        runtime = resolve_runtime_provider(
            requested=self._codex_provider,
            explicit_base_url=None,
            explicit_api_key=None,
            target_model=model_id,
        )
        pool, credential_id = self._runtime_pool_context(runtime)
        api_mode = str(runtime.get("api_mode") or "codex_responses")
        return ResolvedProxyTarget(
            public_id=model_id,
            provider=str(runtime.get("provider") or self._codex_provider),
            model=model_id,
            api_mode=api_mode,
            base_url=str(runtime.get("base_url") or ""),
            api_key=str(runtime.get("api_key") or ""),
            runtime=runtime,
            credential_pool=pool,
            credential_id=credential_id,
        )

    def _codex_responses_payload(self, body: Dict[str, Any], target: ResolvedProxyTarget, *, stream: bool) -> Dict[str, Any]:
        payload = dict(body)
        payload["model"] = target.model
        payload["stream"] = stream
        is_chatgpt_codex_backend = _is_chatgpt_codex_backend(target)
        if ("reasoning_effort" in payload or "model_reasoning_effort" in payload) and "reasoning" not in payload:
            reasoning = _responses_reasoning_from_chat(payload)
            if reasoning:
                payload["reasoning"] = reasoning
        payload.pop("reasoning_effort", None)
        payload.pop("model_reasoning_effort", None)
        for key in _CODEX_RESPONSES_UNSUPPORTED_SDK_PARAMS:
            payload.pop(key, None)
        if is_chatgpt_codex_backend:
            payload["store"] = False
            payload.pop("max_output_tokens", None)
            reasoning = payload.get("reasoning")
            if isinstance(reasoning, dict) and str(reasoning.get("summary") or "").lower() == "none":
                reasoning = dict(reasoning)
                reasoning.pop("summary", None)
                payload["reasoning"] = reasoning
            if not str(payload.get("instructions") or "").strip():
                payload["instructions"] = "You are a helpful assistant."
            if isinstance(payload.get("input"), str):
                payload["input"] = [{"role": "user", "content": payload["input"]}]
            for key in _CODEX_CHATGPT_UNSUPPORTED_SAMPLING_PARAMS:
                payload.pop(key, None)
        return payload

    async def _call_codex_responses_passthrough(
        self,
        body: Dict[str, Any],
        target: ResolvedProxyTarget,
    ) -> Dict[str, Any]:
        payload = self._codex_responses_payload(body, target, stream=False)

        def _call() -> Dict[str, Any]:
            client = self._create_openai_client(target)
            try:
                if _is_chatgpt_codex_backend(target):
                    stream_payload = dict(payload)
                    stream_payload["stream"] = True
                    response = self._collect_streaming_response(client.responses.create(**stream_payload))
                else:
                    response = client.responses.create(**payload)
                return _object_to_dict(response)
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

        try:
            data = await asyncio.get_running_loop().run_in_executor(None, _call)
        except Exception as exc:
            self._mark_target_failure(target, exc)
            raise
        self._record_target_success(target)
        return data

    @staticmethod
    async def _write_sse_event(response: "web.StreamResponse", event: str, data: Any) -> None:
        if event:
            await response.write(f"event: {event}\n".encode())
        await response.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode())

    async def _stream_codex_responses_passthrough(
        self,
        request: "web.Request",
        body: Dict[str, Any],
        target: ResolvedProxyTarget,
    ) -> "web.StreamResponse":
        payload = self._codex_responses_payload(body, target, stream=True)
        loop = asyncio.get_running_loop()
        sentinel = object()
        client = None
        stream = None
        try:
            def _open_stream() -> Any:
                nonlocal client
                client = self._create_openai_client(target)
                return client.responses.create(**payload)

            stream = await loop.run_in_executor(None, _open_stream)
            assert web is not None
            response = web.StreamResponse(status=200, headers=self._sse_headers(request))
            await response.prepare(request)
            iterator = iter(stream)
            while True:
                event = await loop.run_in_executor(None, self._next_stream_item, iterator, sentinel)
                if event is sentinel:
                    break
                data = _object_to_dict(event)
                if not data:
                    continue
                event_type = str(data.get("type") or self._event_attr(event, "type", "") or "")
                await self._write_sse_event(response, event_type, data)
            if self._include_done_event:
                await self._write_sse_done(response)
            self._record_target_success(target)
        except Exception as exc:
            self._mark_target_failure(target, exc)
            raise
        finally:
            for obj in (stream, client):
                close = getattr(obj, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        return response

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
        profile = None
        try:
            from providers import get_provider_profile

            profile = get_provider_profile(target.provider)
        except Exception:
            pass
        headers: Dict[str, str] = {}
        if profile and profile.default_headers:
            headers.update(profile.default_headers)
        if _is_chatgpt_codex_backend(target):
            from agent.auxiliary_client import _codex_cloudflare_headers

            headers.update(_codex_cloudflare_headers(target.api_key))
        if headers:
            kwargs["default_headers"] = headers
        if profile and profile.request_header_prefixes_to_strip:
            from agent.process_bootstrap import (
                build_keepalive_http_client,
                install_provider_request_header_filter,
            )

            http_client = build_keepalive_http_client(target.base_url)
            if http_client is not None:
                install_provider_request_header_filter(
                    http_client,
                    target.provider,
                )
                kwargs["http_client"] = http_client
        return OpenAI(**kwargs)

    @staticmethod
    def _sse_headers(request: "web.Request") -> Dict[str, str]:
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        try:
            origin = request.headers.get("Origin", "")
            adapter = request.app.get("api_server_adapter")
            cors_fn = getattr(adapter, "_cors_headers_for_origin", None)
            if origin and callable(cors_fn):
                cors = cors_fn(origin)
                if cors:
                    headers.update(cors)
        except Exception:
            pass
        return headers

    @staticmethod
    def _chat_stream_chunk(
        public_model: str,
        *,
        delta: Optional[Dict[str, Any]] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        chunk: Dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": public_model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta or {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage is not None:
            chunk["usage"] = usage
        return chunk

    @staticmethod
    async def _write_sse_data(response: "web.StreamResponse", data: Any) -> None:
        await response.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode())

    @staticmethod
    async def _write_sse_done(response: "web.StreamResponse") -> None:
        await response.write(b"data: [DONE]\n\n")

    @staticmethod
    def _next_stream_item(iterator: Any, sentinel: Any) -> Any:
        try:
            return next(iterator)
        except StopIteration:
            return sentinel

    @staticmethod
    def _event_attr(event: Any, name: str, default: Any = None) -> Any:
        if isinstance(event, dict):
            return event.get(name, default)
        return getattr(event, name, default)

    async def _stream_chat_completions(
        self,
        request: "web.Request",
        body: Dict[str, Any],
        target: ResolvedProxyTarget,
        spec: ProxyModelSpec,
    ) -> "web.StreamResponse":
        payload = dict(spec.request_defaults)
        payload.update(body)
        payload["model"] = target.model
        payload["stream"] = True

        response = web.StreamResponse(status=200, headers=self._sse_headers(request))
        await response.prepare(request)
        loop = asyncio.get_running_loop()
        sentinel = object()
        client = None
        stream = None
        try:
            def _open_stream() -> Any:
                nonlocal client
                client = self._create_openai_client(target)
                return client.chat.completions.create(**payload)

            stream = await loop.run_in_executor(None, _open_stream)
            iterator = iter(stream)
            while True:
                chunk_obj = await loop.run_in_executor(None, self._next_stream_item, iterator, sentinel)
                if chunk_obj is sentinel:
                    break
                chunk = _object_to_dict(chunk_obj)
                if not chunk:
                    continue
                chunk["model"] = spec.public_id
                await self._write_sse_data(response, chunk)
            await self._write_sse_done(response)
        finally:
            for obj in (stream, client):
                close = getattr(obj, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        return response

    async def _stream_codex_chat_compat(
        self,
        request: "web.Request",
        body: Dict[str, Any],
        target: ResolvedProxyTarget,
        spec: ProxyModelSpec,
    ) -> "web.StreamResponse":
        from agent.codex_responses_adapter import _chat_messages_to_responses_input, _responses_tools
        from agent.transports.codex import ResponsesApiTransport

        messages = body.get("messages") or []
        instructions = "\n".join(
            str(msg.get("content") or "")
            for msg in messages
            if isinstance(msg, dict) and msg.get("role") in {"system", "developer"}
        ).strip()
        if not instructions:
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
            "stream": True,
        }
        if instructions:
            payload["instructions"] = instructions
        tools = _responses_tools(body.get("tools"))
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = _responses_tool_choice_from_chat(body.get("tool_choice", "auto"))
        is_chatgpt_codex_backend = _is_chatgpt_codex_backend(target)
        reasoning = _responses_reasoning_from_chat(body)
        if reasoning:
            payload["reasoning"] = reasoning
        for passthrough_key in (
            "parallel_tool_calls",
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "logprobs",
            "top_logprobs",
            "include",
            "prompt_cache_key",
            "service_tier",
        ):
            if body.get(passthrough_key) is not None and _should_pass_codex_request_param(
                passthrough_key,
                is_chatgpt_codex_backend=is_chatgpt_codex_backend,
            ):
                payload[passthrough_key] = body.get(passthrough_key)
        if not is_chatgpt_codex_backend and body.get("max_tokens") is not None:
            payload["max_output_tokens"] = body.get("max_tokens")
        if not is_chatgpt_codex_backend and body.get("max_completion_tokens") is not None:
            payload["max_output_tokens"] = body.get("max_completion_tokens")

        response = web.StreamResponse(status=200, headers=self._sse_headers(request))
        await response.prepare(request)
        await self._write_sse_data(response, self._chat_stream_chunk(spec.public_id, delta={"role": "assistant"}))

        loop = asyncio.get_running_loop()
        sentinel = object()
        client = None
        stream = None
        terminal_response = None
        collected_output_items: List[Any] = []
        collected_text_deltas: List[str] = []
        try:
            def _open_stream() -> Any:
                nonlocal client
                client = self._create_openai_client(target)
                return client.responses.create(**payload)

            def _item_type(item: Any) -> str:
                return str(self._event_attr(item, "type", "") or "")

            def _tool_item_identity(item: Any) -> tuple[Optional[str], Optional[str], str, str]:
                item_id = self._event_attr(item, "id")
                call_id = self._event_attr(item, "call_id") or item_id
                name = self._event_attr(item, "name", "") or ""
                arguments = self._event_attr(item, "arguments", "")
                if arguments is None:
                    arguments = ""
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                return (
                    str(item_id) if item_id else None,
                    str(call_id) if call_id else None,
                    str(name),
                    arguments,
                )

            stream = await loop.run_in_executor(None, _open_stream)
            iterator = iter(stream)
            streamed_tool_calls = False
            tool_call_indexes: Dict[str, int] = {}
            next_tool_call_index = 0
            while True:
                event = await loop.run_in_executor(None, self._next_stream_item, iterator, sentinel)
                if event is sentinel:
                    break
                event_type = self._event_attr(event, "type")
                if event_type == "response.output_text.delta":
                    delta = self._event_attr(event, "delta", "") or ""
                    if delta:
                        collected_text_deltas.append(str(delta))
                        await self._write_sse_data(
                            response,
                            self._chat_stream_chunk(spec.public_id, delta={"content": str(delta)}),
                        )
                elif event_type == "response.output_item.added":
                    item = self._event_attr(event, "item")
                    if item is not None and _item_type(item) in {"function_call", "custom_tool_call"}:
                        item_id, call_id, name, arguments = _tool_item_identity(item)
                        index = next_tool_call_index
                        next_tool_call_index += 1
                        for key in (item_id, call_id):
                            if key:
                                tool_call_indexes[key] = index
                        streamed_tool_calls = True
                        await self._write_sse_data(
                            response,
                            self._chat_stream_chunk(
                                spec.public_id,
                                delta={
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": call_id or f"call_{uuid.uuid4().hex[:16]}",
                                            "type": "function",
                                            "function": {"name": name, "arguments": arguments},
                                        }
                                    ]
                                },
                            ),
                        )
                elif event_type in {
                    "response.function_call_arguments.delta",
                    "response.custom_tool_call_input.delta",
                    "response.output_item.arguments.delta",
                }:
                    delta = self._event_attr(event, "delta", "") or self._event_attr(event, "input", "") or ""
                    item_id = self._event_attr(event, "item_id") or self._event_attr(event, "output_item_id")
                    call_id = self._event_attr(event, "call_id")
                    index = None
                    for key in (item_id, call_id):
                        if key in tool_call_indexes:
                            index = tool_call_indexes[key]
                            break
                    if index is None:
                        index = next_tool_call_index
                        next_tool_call_index += 1
                        for key in (item_id, call_id):
                            if key:
                                tool_call_indexes[str(key)] = index
                    if delta:
                        streamed_tool_calls = True
                        await self._write_sse_data(
                            response,
                            self._chat_stream_chunk(
                                spec.public_id,
                                delta={"tool_calls": [{"index": index, "function": {"arguments": str(delta)}}]},
                            ),
                        )
                elif event_type == "response.output_item.done":
                    item = self._event_attr(event, "item")
                    if item is not None:
                        collected_output_items.append(item)
                        if _item_type(item) in {"function_call", "custom_tool_call"}:
                            item_id, call_id, name, arguments = _tool_item_identity(item)
                            known_index = None
                            for key in (item_id, call_id):
                                if key in tool_call_indexes:
                                    known_index = tool_call_indexes[key]
                                    break
                            if known_index is None:
                                known_index = next_tool_call_index
                                next_tool_call_index += 1
                                streamed_tool_calls = True
                                await self._write_sse_data(
                                    response,
                                    self._chat_stream_chunk(
                                        spec.public_id,
                                        delta={
                                            "tool_calls": [
                                                {
                                                    "index": known_index,
                                                    "id": call_id or f"call_{uuid.uuid4().hex[:16]}",
                                                    "type": "function",
                                                    "function": {"name": name, "arguments": arguments},
                                                }
                                            ]
                                        },
                                    ),
                                )
                            for key in (item_id, call_id):
                                if key:
                                    tool_call_indexes[key] = known_index
                elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
                    terminal_response = self._event_attr(event, "response")
                    break

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
            normalized = ResponsesApiTransport().normalize_response(terminal_response)
            finish_reason = getattr(normalized, "finish_reason", None) or "stop"
            tool_calls = getattr(normalized, "tool_calls", None)
            if tool_calls and not streamed_tool_calls:
                await self._write_sse_data(
                    response,
                    self._chat_stream_chunk(
                        spec.public_id,
                        delta={
                            "tool_calls": [
                                _openai_tool_call_from_normalized(tc, index=idx)
                                for idx, tc in enumerate(tool_calls)
                            ]
                        },
                    ),
                )
            await self._write_sse_data(
                response,
                self._chat_stream_chunk(
                    spec.public_id,
                    finish_reason=finish_reason,
                    usage=_usage_to_openai_dict(terminal_response),
                ),
            )
            await self._write_sse_done(response)
        finally:
            for obj in (stream, client):
                close = getattr(obj, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        return response

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
            payload["tool_choice"] = _responses_tool_choice_from_chat(body.get("tool_choice", "auto"))
        is_chatgpt_codex_backend = _is_chatgpt_codex_backend(target)
        reasoning = _responses_reasoning_from_chat(body)
        if reasoning:
            payload["reasoning"] = reasoning
        for passthrough_key in (
            "parallel_tool_calls",
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "logprobs",
            "top_logprobs",
            "include",
            "prompt_cache_key",
            "service_tier",
        ):
            if body.get(passthrough_key) is not None and _should_pass_codex_request_param(
                passthrough_key,
                is_chatgpt_codex_backend=is_chatgpt_codex_backend,
            ):
                payload[passthrough_key] = body.get(passthrough_key)
        # ChatGPT's Codex backend rejects max_output_tokens on its streaming
        # compatibility endpoint, while standard Responses-compatible backends
        # and tests expect the Chat Completions token limit to be mapped.
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
