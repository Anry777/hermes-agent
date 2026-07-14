import json
import time
from unittest.mock import patch


def _write_auth_store(path, payload):
    home = path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps(payload))
    return home


def test_codex_aux_does_not_bypass_exhausted_pool(monkeypatch):
    from agent.auxiliary_client import _read_codex_access_token

    with (
        patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)),
        patch("hermes_cli.auth._read_codex_tokens", return_value={
            "tokens": {"access_token": "singleton-token", "refresh_token": "refresh"}
        }),
    ):
        assert _read_codex_access_token() is None


def test_available_entries_clears_stale_reset_at_on_ok_entry(tmp_path, monkeypatch):
    home = _write_auth_store(tmp_path, {
        "version": 1,
        "credential_pool": {
            "openrouter": [{
                "id": "cred-1",
                "label": "one",
                "auth_type": "api_key",
                "source": "manual",
                "access_token": "sk-one",
                "last_status": "ok",
                "last_error_reset_at": time.time() + 3600,
            }]
        },
    })
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.credential_pool import load_pool

    selected = load_pool("openrouter").select()
    assert selected is not None
    assert selected.id == "cred-1"
    stored = json.loads((home / "auth.json").read_text())
    assert stored["credential_pool"]["openrouter"][0].get("last_error_reset_at") is None


def test_dead_credential_does_not_keep_retry_reset_at(tmp_path, monkeypatch):
    home = _write_auth_store(tmp_path, {
        "version": 1,
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "dead",
                    "label": "dead",
                    "auth_type": "oauth",
                    "source": "manual:device_code",
                    "access_token": "tok-dead",
                },
                {
                    "id": "live",
                    "label": "live",
                    "auth_type": "oauth",
                    "source": "manual:device_code",
                    "access_token": "tok-live",
                },
            ]
        },
    })
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.credential_pool import STATUS_DEAD, load_pool

    pool = load_pool("openai-codex")
    selected = pool.select()
    assert selected is not None
    assert selected.id == "dead"
    pool.mark_exhausted_and_rotate(
        status_code=401,
        error_context={"reason": "token_invalidated", "reset_at": time.time() + 3600},
    )
    stored = json.loads((home / "auth.json").read_text())
    dead = next(e for e in stored["credential_pool"]["openai-codex"] if e["id"] == "dead")
    assert dead["last_status"] == STATUS_DEAD
    assert dead.get("last_error_reset_at") is None
