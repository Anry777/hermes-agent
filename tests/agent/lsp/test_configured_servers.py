"""Tests for config-driven LSP server registration."""
from __future__ import annotations

from pathlib import Path

from agent.lsp.manager import LSPService
from agent.lsp.servers import ServerDef, build_servers_from_lsp_config, find_server_for_file, language_id_for


def _bsl_lsp_config(url: str = "ws://127.0.0.1:8025/lsp") -> dict:
    return {
        "enabled": True,
        "servers": {
            "bsl-language-server": {
                "language_id": "bsl",
                "extensions": [".bsl"],
                "transport": {"type": "websocket", "url": url},
                "workspace_root": "git",
            }
        },
    }


def test_custom_server_config_registers_extension_and_language_id():
    servers, language_by_ext = build_servers_from_lsp_config(_bsl_lsp_config())

    srv = find_server_for_file("/repo/src/module.bsl", servers=servers)

    assert srv is not None
    assert srv.server_id == "bsl-language-server"
    assert srv.configured is True
    assert srv.transport_type == "websocket"
    assert srv.transport_target == "ws://127.0.0.1:8025/lsp"
    assert language_id_for("/repo/src/module.bsl", language_by_ext=language_by_ext) == "bsl"


def test_legacy_custom_servers_alias_is_supported():
    cfg = {
        "custom_servers": {
            "bsl": {
                "server_id": "bsl-language-server",
                "language_id": "bsl",
                "extensions": ["bsl"],
                "transport": "websocket",
                "url": "ws://127.0.0.1:8025/lsp",
            }
        }
    }

    servers, language_by_ext = build_servers_from_lsp_config(cfg)

    srv = find_server_for_file("/repo/module.bsl", servers=servers)
    assert srv is not None
    assert srv.server_id == "bsl-language-server"
    assert srv.extensions == (".bsl",)
    assert language_id_for("/repo/module.bsl", language_by_ext=language_by_ext) == "bsl"


def test_configured_websocket_server_does_not_require_binary_install():
    servers, language_by_ext = build_servers_from_lsp_config(_bsl_lsp_config())
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=0.2,
        install_strategy="manual",
        servers=servers,
        language_by_ext=language_by_ext,
    )
    try:
        info = svc.get_status()
    finally:
        svc.shutdown()

    configured = {s["server_id"]: s for s in info["servers"]}
    assert configured["bsl-language-server"]["source"] == "configured"
    assert configured["bsl-language-server"]["transport"] == "websocket"
    assert configured["bsl-language-server"]["availability"] == "configured"


def test_invalid_websocket_url_is_controlled_unavailable(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    bsl = repo / "module.bsl"
    bsl.write_text("Процедура Тест()\nКонецПроцедуры\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    servers, language_by_ext = build_servers_from_lsp_config(_bsl_lsp_config("http://127.0.0.1/lsp"))
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=0.2,
        install_strategy="manual",
        servers=servers,
        language_by_ext=language_by_ext,
    )
    try:
        assert svc.enabled_for(str(bsl))
        assert svc.get_diagnostics_sync(str(bsl), timeout=0.5) == []
        info = svc.get_status()
    finally:
        svc.shutdown()

    configured = {s["server_id"]: s for s in info["servers"]}
    assert configured["bsl-language-server"]["availability"] == "unavailable"
    assert "invalid websocket url" in configured["bsl-language-server"]["availability_reason"]


def test_lsp_which_prints_websocket_target(monkeypatch, capsys):
    from agent.lsp import cli

    servers, _ = build_servers_from_lsp_config(_bsl_lsp_config())
    monkeypatch.setattr(cli, "_configured_registry", lambda: servers)

    rc = cli._cmd_which("bsl-language-server")

    out = capsys.readouterr()
    assert rc == 0
    assert out.out.strip() == "websocket ws://127.0.0.1:8025/lsp"
    assert out.err == ""


def test_lsp_list_marks_websocket_server_configured(monkeypatch, capsys):
    from agent.lsp import cli

    servers, _ = build_servers_from_lsp_config(_bsl_lsp_config())
    configured = [s for s in servers if s.server_id == "bsl-language-server"]
    monkeypatch.setattr(cli, "_configured_registry", lambda: configured)

    rc = cli._cmd_list(installed_only=False)

    out = capsys.readouterr()
    assert rc == 0
    assert "bsl-language-server" in out.out
    assert "[configured]" in out.out
    assert ".bsl" in out.out


def test_lsp_test_rejects_stdio_server_without_binary(monkeypatch, capsys):
    from agent.lsp import cli

    stdio_server = ServerDef(
        server_id="example-stdio",
        extensions=(".ex",),
        resolve_root=lambda fp, ws: ws,
        build_spawn=lambda root, ctx: None,
        description="example",
    )
    monkeypatch.setattr(cli, "_configured_registry", lambda: [stdio_server])

    rc = cli._cmd_test("example-stdio")

    out = capsys.readouterr()
    assert rc == 1
    assert "only configured websocket servers" in out.err
