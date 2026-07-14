"""Tests for VibeMode auth and runtime provider wiring."""

from __future__ import annotations

import json


def test_auth_add_vibemode_api_key_resolves_dynamic_runtime(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("VIBEMODE_API_KEY", raising=False)
    monkeypatch.delenv("VIBEMODE_BASE_URL", raising=False)

    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
    from hermes_cli.auth_commands import auth_add_command
    from hermes_cli.runtime_provider import resolve_runtime_provider

    assert resolve_provider("vibemode") == "vibemode"
    assert "vibemode" in PROVIDER_REGISTRY

    class _Args:
        provider = "vibemode"
        auth_type = "api-key"
        api_key = "vm-test-key"
        label = "vm"

    auth_add_command(_Args())

    payload = json.loads((hermes_home / "auth.json").read_text())
    entry = payload["credential_pool"]["vibemode"][0]
    assert entry["label"] == "vm"
    assert entry["access_token"] == "vm-test-key"
    assert entry["base_url"] == "https://api.vibemod.pro/v1"

    runtime = resolve_runtime_provider(requested="vibemode", target_model="gpt-5.5")
    assert runtime["provider"] == "vibemode"
    assert runtime["api_mode"] == "codex_responses"
    assert runtime["base_url"] == "https://api.vibemod.pro/v1"
    assert runtime["api_key"] == "vm-test-key"


def test_vibemode_routes_known_models_to_their_api_surfaces(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli.auth_commands import auth_add_command
    from hermes_cli.runtime_provider import resolve_runtime_provider

    class _Args:
        provider = "vibemode"
        auth_type = "api-key"
        api_key = "vm-test-key"
        label = "vm"

    auth_add_command(_Args())

    assert resolve_runtime_provider(
        requested="vibemode", target_model="deepseek-v4-pro"
    )["api_mode"] == "chat_completions"
    assert resolve_runtime_provider(
        requested="vibemode", target_model="gpt-5.4-mini"
    )["api_mode"] == "codex_responses"
    assert resolve_runtime_provider(
        requested="vibemode", target_model="minimax-m3"
    )["api_mode"] == "anthropic_messages"


def test_vibemode_unknown_model_uses_provider_default_not_catalog_overlay(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli.auth_commands import auth_add_command
    from hermes_cli.runtime_provider import resolve_runtime_provider

    class _Args:
        provider = "vibemode"
        auth_type = "api-key"
        api_key = "vm-test-key"
        label = "vm"

    auth_add_command(_Args())

    runtime = resolve_runtime_provider(requested="vibemode", target_model="new-live-model")
    assert runtime["api_mode"] == "chat_completions"
