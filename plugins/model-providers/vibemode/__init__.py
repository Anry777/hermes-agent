"""VibeMode provider profile."""

from __future__ import annotations

from providers import register_provider
from providers.base import ProviderProfile


VIBEMODE_MODEL_PROPERTIES = {
    # Chat Completions-first models. VibeMode also exposes /v1/responses for
    # these, but the provider marks their primary API as Chat.
    "deepseek-v4-flash": {
        "context_length": 1_000_000,
        "api_mode": "chat_completions",
        "endpoint_kinds": ("chat_completions", "responses"),
        "capabilities": ("chat", "reasoning", "tools"),
    },
    "deepseek-v4-pro": {
        "context_length": 1_000_000,
        "api_mode": "chat_completions",
        "endpoint_kinds": ("chat_completions", "responses"),
        "capabilities": ("chat", "reasoning", "tools"),
    },
    "glm-5.1": {
        "context_length": 203_000,
        "api_mode": "chat_completions",
        "endpoint_kinds": ("chat_completions", "responses"),
        "capabilities": ("chat", "reasoning", "tools"),
    },
    "glm-5.2": {
        "context_length": 203_000,
        "api_mode": "chat_completions",
        "endpoint_kinds": ("chat_completions", "responses"),
        "capabilities": ("chat", "reasoning", "tools"),
    },
    "kimi-k2.6": {
        "context_length": 262_000,
        "api_mode": "chat_completions",
        "endpoint_kinds": ("chat_completions", "responses"),
        "capabilities": ("chat", "vision", "reasoning"),
    },
    "mimo-v2.5": {
        "context_length": 1_000_000,
        "api_mode": "chat_completions",
        "endpoint_kinds": ("chat_completions", "responses"),
        "capabilities": ("chat", "reasoning", "tools"),
    },
    "mimo-v2.5-pro": {
        "context_length": 1_000_000,
        "api_mode": "chat_completions",
        "endpoint_kinds": ("chat_completions", "responses"),
        "capabilities": ("chat", "reasoning", "tools"),
    },
    # Responses-first GPT models.
    "gpt-5.4": {
        "context_length": 272_000,
        "api_mode": "codex_responses",
        "endpoint_kinds": ("responses", "chat_completions"),
        "capabilities": ("chat", "vision", "reasoning"),
    },
    "gpt-5.4-mini": {
        "context_length": 272_000,
        "api_mode": "codex_responses",
        "endpoint_kinds": ("responses", "chat_completions"),
        "capabilities": ("chat", "vision", "reasoning"),
    },
    "gpt-5.5": {
        "context_length": 272_000,
        "api_mode": "codex_responses",
        "endpoint_kinds": ("responses", "chat_completions"),
        "capabilities": ("chat", "vision", "reasoning"),
    },
    # Anthropic Messages-first models.
    "minimax-m3": {
        "context_length": 1_000_000,
        "api_mode": "anthropic_messages",
        "endpoint_kinds": ("messages",),
        "capabilities": ("chat", "vision", "reasoning"),
    },
    "qwen3.7-max": {
        "context_length": 1_000_000,
        "api_mode": "anthropic_messages",
        "endpoint_kinds": ("messages",),
        "capabilities": ("chat", "reasoning", "tools"),
    },
    "qwen3.7-plus": {
        "context_length": 1_000_000,
        "api_mode": "anthropic_messages",
        "endpoint_kinds": ("messages",),
        "capabilities": ("chat", "reasoning", "tools"),
    },
}


vibemode = ProviderProfile(
    name="vibemode",
    display_name="VibeMode",
    description="Multi-model gateway with Chat, Responses, and Messages endpoints",
    signup_url="https://vibemod.pro/",
    api_mode="chat_completions",
    env_vars=("VIBEMODE_API_KEY", "VIBEMODE_BASE_URL"),
    base_url="https://api.vibemod.pro/v1",
    auth_type="api_key",
    # Cloudflare in front of VibeMode blocks the OpenAI Python SDK's default
    # ``OpenAI/Python ...`` User-Agent on /v1/responses. Send a stable Hermes UA
    # from the provider profile instead of relying on SDK defaults.
    default_headers={"User-Agent": "HermesAgent/1.0"},
    # Some VibeMode credentials reject OpenAI SDK identity metadata as an
    # authentication error even when the same key and payload work over raw HTTP.
    request_header_prefixes_to_strip=("x-stainless-",),
    # The model catalog is intentionally dynamic via /v1/models.  These maps are
    # metadata overlays for known VibeMode slugs, not the source of truth for the
    # list of available models.
    model_context_lengths={
        model: int(props["context_length"])
        for model, props in VIBEMODE_MODEL_PROPERTIES.items()
    },
    model_api_modes={
        model: str(props["api_mode"])
        for model, props in VIBEMODE_MODEL_PROPERTIES.items()
    },
    model_capabilities={
        model: tuple(props["capabilities"])
        for model, props in VIBEMODE_MODEL_PROPERTIES.items()
    },
    model_endpoint_kinds={
        model: tuple(props["endpoint_kinds"])
        for model, props in VIBEMODE_MODEL_PROPERTIES.items()
    },
)

register_provider(vibemode)
