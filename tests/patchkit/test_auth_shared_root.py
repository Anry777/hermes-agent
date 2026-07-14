import json
from pathlib import Path


def test_profile_auth_reads_shared_root_even_when_profile_auth_exists(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    (root / "auth.json").write_text(json.dumps({
        "version": 1,
        "active_provider": "nous",
        "providers": {"nous": {"access_token": "root-token"}},
    }))
    (profile / "auth.json").write_text(json.dumps({
        "version": 1,
        "active_provider": "nous",
        "providers": {"nous": {"access_token": "profile-token"}},
    }))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    from hermes_cli.auth import _auth_file_path, _auth_lock_path, _load_auth_store

    assert _auth_file_path() == root / "auth.json"
    assert _auth_lock_path() == root / "auth.lock"
    assert _load_auth_store()["providers"]["nous"]["access_token"] == "root-token"


def test_profile_auth_writes_shared_root_not_profile_store(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    root_payload = {
        "version": 1,
        "active_provider": "anthropic",
        "providers": {"anthropic": {"access_token": "root-token"}},
        "credential_pool": {
            "anthropic": [{"id": "a", "auth_type": "oauth", "source": "manual"}],
            "openrouter": [{"id": "b", "auth_type": "api_key", "source": "manual"}],
        },
    }
    profile_payload = {
        "version": 1,
        "active_provider": "anthropic",
        "providers": {"anthropic": {"access_token": "profile-token"}},
    }
    (root / "auth.json").write_text(json.dumps(root_payload))
    (profile / "auth.json").write_text(json.dumps(profile_payload))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    from hermes_cli.auth import clear_provider_auth

    assert clear_provider_auth("anthropic") is True
    root_after = json.loads((root / "auth.json").read_text())
    assert "anthropic" not in root_after.get("providers", {})
    assert "anthropic" not in root_after.get("credential_pool", {})
    assert "openrouter" in root_after.get("credential_pool", {})
    assert json.loads((profile / "auth.json").read_text()) == profile_payload


def test_auxiliary_nous_reads_shared_root_from_profile(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    (root / "auth.json").write_text(json.dumps({
        "active_provider": "nous",
        "providers": {"nous": {
            "access_token": "root-token",
            "agent_key": "root-agent",
            "inference_base_url": "https://inference.example/v1",
        }},
    }))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    from agent.auxiliary_client import _read_nous_auth

    provider = _read_nous_auth()
    assert provider is not None
    assert provider["access_token"] == "root-token"
    assert provider["agent_key"] == "root-agent"
