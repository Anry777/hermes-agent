"""``hermes lsp`` CLI subcommand.

Subcommands:

- ``status`` — show service state, configured servers, install status.
- ``install <server_id>`` — eagerly install one server's binary.
- ``install-all`` — try to install every server with a known recipe.
- ``restart`` — tear down running clients so the next edit re-spawns.
- ``which <server_id>`` — print the resolved binary path for one server.
- ``list`` — print the registry of supported servers.

The handlers are kept here (rather than in
``hermes_cli/main.py``) so the LSP module ships self-contained.
"""
from __future__ import annotations

import argparse
import sys


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Wire the ``hermes lsp`` subcommand tree into the main argparse."""
    parser = subparsers.add_parser(
        "lsp",
        help="Language Server Protocol management",
        description=(
            "Manage the LSP layer that powers post-write semantic "
            "diagnostics in write_file/patch."
        ),
    )
    sub = parser.add_subparsers(dest="lsp_command")

    sub_status = sub.add_parser("status", help="Show LSP service status")
    sub_status.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    sub_status.add_argument(
        "--check", action="store_true", help="Health-check configured websocket servers"
    )

    sub_list = sub.add_parser("list", help="List supported language servers")
    sub_list.add_argument(
        "--installed-only",
        action="store_true",
        help="Only show servers whose binary is currently available",
    )

    sub_install = sub.add_parser("install", help="Install a server binary")
    sub_install.add_argument("server", help="Server id (e.g. pyright, gopls)")

    sub_install_all = sub.add_parser(
        "install-all",
        help="Install every server with a known auto-install recipe",
    )
    sub_install_all.add_argument(
        "--include-manual",
        action="store_true",
        help="Even attempt servers marked manual-install (best effort)",
    )

    sub_restart = sub.add_parser(
        "restart",
        help="Tear down running LSP clients (next edit re-spawns)",
    )

    sub_which = sub.add_parser("which", help="Print binary path for a server")
    sub_which.add_argument("server", help="Server id")

    sub_test = sub.add_parser("test", help="Health-check a configured LSP server")
    sub_test.add_argument("server", help="Server id")

    parser.set_defaults(func=run_lsp_command)


def run_lsp_command(args: argparse.Namespace) -> int:
    """Top-level dispatcher for ``hermes lsp <subcommand>``."""
    sub = getattr(args, "lsp_command", None) or "status"
    try:
        if sub == "status":
            return _cmd_status(getattr(args, "json", False), getattr(args, "check", False))
        if sub == "list":
            return _cmd_list(getattr(args, "installed_only", False))
        if sub == "install":
            return _cmd_install(args.server)
        if sub == "install-all":
            return _cmd_install_all(getattr(args, "include_manual", False))
        if sub == "restart":
            return _cmd_restart()
        if sub == "which":
            return _cmd_which(args.server)
        if sub == "test":
            return _cmd_test(args.server)
        sys.stderr.write(f"unknown lsp subcommand: {sub}\n")
        return 2
    except KeyboardInterrupt:
        return 130


def _load_lsp_cfg() -> dict:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:  # noqa: BLE001
        return {}
    lsp_cfg = (cfg.get("lsp") or {}) if isinstance(cfg, dict) else {}
    return lsp_cfg if isinstance(lsp_cfg, dict) else {}


def _configured_registry():
    from agent.lsp.servers import build_servers_from_lsp_config
    return build_servers_from_lsp_config(_load_lsp_cfg())[0]


def _cmd_status(emit_json: bool, check: bool = False) -> int:
    from agent.lsp import get_service
    from agent.lsp.install import detect_status
    from agent.lsp.servers import server_availability

    svc = get_service()
    service_active = svc is not None
    info = svc.get_status() if svc is not None else {"enabled": False, "servers": []}
    registry = info.get("servers") or []
    if not registry:
        registry = []
        for s in _configured_registry():
            availability, reason = server_availability(s)
            registry.append(
                {
                    "server_id": s.server_id,
                    "extensions": list(s.extensions),
                    "description": s.description,
                    "source": "configured" if s.configured else "builtin",
                    "transport": s.transport_type,
                    "target": s.transport_target,
                    "availability": availability,
                    "availability_reason": reason,
                    "binary_status": detect_status(_recipe_pkg_for(s.server_id)) if s.transport_type != "websocket" else availability,
                }
            )

    if emit_json:
        import json
        payload = {"service": info, "registry": registry}
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0

    out = []
    out.append("LSP Service")
    out.append("===========")
    out.append(f"  enabled:         {info.get('enabled', False)}")
    if service_active:
        out.append(f"  wait_mode:       {info.get('wait_mode')}")
        out.append(f"  wait_timeout:    {info.get('wait_timeout')}s")
        out.append(f"  install_strategy:{info.get('install_strategy')}")
        clients = info.get("clients") or []
        if clients:
            out.append(f"  active clients:  {len(clients)}")
            for c in clients:
                out.append(
                    f"    - {c['server_id']:20s} state={c['state']:10s} root={c['workspace_root']}"
                )
        else:
            out.append("  active clients:  none")
        broken = info.get("broken") or []
        if broken:
            out.append(f"  broken pairs:    {len(broken)}")
            for b in broken:
                out.append(f"    - {b}")
        disabled = info.get("disabled_servers") or []
        if disabled:
            out.append(f"  disabled in cfg: {', '.join(disabled)}")

    # Surface backend-tool gaps that aren't visible in the registry table:
    # some servers spawn fine but emit no diagnostics without a sidecar
    # binary (bash-language-server -> shellcheck).
    backend_warnings = _backend_warnings()
    if backend_warnings:
        out.append("")
        out.append("Backend warnings")
        out.append("================")
        for line in backend_warnings:
            out.append(f"  ! {line}")
    out.append("")
    out.append("Registered Servers")
    out.append("==================")
    for s in registry:
        server_id = s["server_id"]
        extensions = list(s.get("extensions") or [])
        transport = s.get("transport") or "stdio"
        status = s.get("availability") or s.get("binary_status") or "unknown"
        if transport != "websocket" and "binary_status" not in s:
            status = detect_status(_recipe_pkg_for(server_id))
        marker = {
            "installed": "✓",
            "configured": "✓",
            "builtin": "·",
            "missing": "·",
            "manual-only": "?",
            "unavailable": "!",
        }.get(status, " ")
        ext_summary = ", ".join(extensions[:5])
        if len(extensions) > 5:
            ext_summary += f", … (+{len(extensions) - 5})"
        source = s.get("source") or "builtin"
        out.append(
            f"  {marker} {server_id:24s} [{status}] {ext_summary}"
        )
        if transport != "stdio":
            out.append(f"      {source}; {transport} {s.get('target') or ''}".rstrip())
        if s.get("description"):
            out.append(f"      {s['description']}")
        if status == "unavailable" and s.get("availability_reason"):
            out.append(f"      unavailable: {s['availability_reason']}")
    if check:
        out.append("")
        out.append("Health checks")
        out.append("=============")
        for s in registry:
            if s.get("transport") != "websocket":
                continue
            rc, lines = _healthcheck_server_dict(s)
            prefix = "ok" if rc == 0 else "failed"
            out.append(f"  {s['server_id']}: {prefix}")
            out.extend(f"    {line}" for line in lines)
    sys.stdout.write("\n".join(out) + "\n")
    return 0


def _cmd_list(installed_only: bool) -> int:
    from agent.lsp.install import detect_status
    from agent.lsp.servers import server_availability

    for s in _configured_registry():
        availability, _reason = server_availability(s)
        status = availability if s.transport_type == "websocket" else detect_status(_recipe_pkg_for(s.server_id))
        if installed_only and status not in {"installed", "configured"}:
            continue
        sys.stdout.write(
            f"{s.server_id:24s} [{status}] {','.join(s.extensions)}\n"
        )
    return 0


def _cmd_install(server_id: str) -> int:
    from agent.lsp.install import try_install, INSTALL_RECIPES, detect_status
    pkg = _recipe_pkg_for(server_id)
    pre_status = detect_status(pkg)
    if pre_status == "installed":
        sys.stdout.write(f"{server_id} already installed\n")
        return 0
    sys.stdout.write(f"installing {server_id} (pkg={pkg}) ...\n")
    sys.stdout.flush()
    bin_path = try_install(pkg, "auto")
    if bin_path is None:
        recipe = INSTALL_RECIPES.get(pkg)
        if recipe and recipe.get("strategy") == "manual":
            sys.stderr.write(
                f"{server_id}: this server requires a manual install. "
                f"See documentation.\n"
            )
        else:
            sys.stderr.write(f"{server_id}: install failed (see logs).\n")
        return 1
    sys.stdout.write(f"installed: {bin_path}\n")
    return 0


def _cmd_install_all(include_manual: bool) -> int:
    from agent.lsp.servers import SERVERS
    from agent.lsp.install import try_install, INSTALL_RECIPES, detect_status

    rc = 0
    for s in SERVERS:
        pkg = _recipe_pkg_for(s.server_id)
        recipe = INSTALL_RECIPES.get(pkg)
        if recipe is None:
            continue
        if recipe.get("strategy") == "manual" and not include_manual:
            continue
        if detect_status(pkg) == "installed":
            sys.stdout.write(f"  {s.server_id:24s} already installed\n")
            continue
        sys.stdout.write(f"  installing {s.server_id} (pkg={pkg}) ... ")
        sys.stdout.flush()
        path = try_install(pkg, "auto")
        if path:
            sys.stdout.write(f"ok ({path})\n")
        else:
            sys.stdout.write("FAILED\n")
            rc = 1
    return rc


def _cmd_restart() -> int:
    from agent.lsp import shutdown_service

    shutdown_service()
    sys.stdout.write("LSP service shut down. Next edit will respawn clients.\n")
    return 0


def _cmd_which(server_id: str) -> int:
    from agent.lsp.install import INSTALL_RECIPES, _existing_binary

    for srv in _configured_registry():
        if srv.server_id == server_id and srv.transport_type == "websocket":
            if srv.transport_target:
                sys.stdout.write(f"websocket {srv.transport_target}\n")
                return 0
            sys.stderr.write(f"{server_id}: websocket url not configured\n")
            return 1

    recipe = INSTALL_RECIPES.get(server_id)
    bin_name = (recipe or {}).get("bin", server_id)
    resolved = _existing_binary(bin_name)
    if resolved:
        sys.stdout.write(resolved + "\n")
        return 0
    sys.stderr.write(f"{server_id}: not installed\n")
    return 1


def _cmd_test(server_id: str) -> int:
    registry = _configured_registry()
    for srv in registry:
        if srv.server_id == server_id:
            rc, lines = _healthcheck_server_def(srv)
            for line in lines:
                stream = sys.stdout if rc == 0 else sys.stderr
                stream.write(line + "\n")
            return rc
    sys.stderr.write(f"{server_id}: unknown LSP server\n")
    return 2


def _healthcheck_server_dict(srv: dict) -> tuple[int, list[str]]:
    target = srv.get("target")
    if srv.get("transport") != "websocket" or not target:
        return 1, ["only configured websocket servers can be health-checked"]
    return _healthcheck_websocket(str(srv.get("server_id") or "lsp"), str(target))


def _healthcheck_server_def(srv) -> tuple[int, list[str]]:
    if srv.transport_type != "websocket" or not srv.transport_target:
        return 1, ["only configured websocket servers can be health-checked"]
    return _healthcheck_websocket(srv.server_id, srv.transport_target)


def _healthcheck_websocket(server_id: str, url: str) -> tuple[int, list[str]]:
    import asyncio
    from agent.lsp.client import LSPClient
    from agent.lsp.transports import WebSocketLSPTransport

    async def _run() -> tuple[int, list[str]]:
        client = LSPClient(
            server_id=server_id,
            workspace_root="/",
            transport=WebSocketLSPTransport(server_id, url),
        )
        try:
            await client.start()
            caps = (client.initialize_result or {}).get("capabilities")
            if not isinstance(caps, dict):
                await client.shutdown()
                return 1, ["connected", "initialize: failed", "capabilities missing"]
            await client.shutdown()
            return 0, ["connected", "initialize: ok", "capabilities received"]
        except Exception as e:  # noqa: BLE001
            try:
                await client.shutdown()
            except Exception:  # noqa: BLE001
                pass
            return 1, [f"connection failed: {type(e).__name__}: {e}"]

    return asyncio.run(_run())


def _recipe_pkg_for(server_id: str) -> str:
    """Map a registry ``server_id`` to its install-recipe package key."""
    # The mapping lives here (not in install.py) because it's a CLI
    # convenience layer.  Most server_ids are also their own recipe
    # key, but a few differ (e.g. ``vue-language-server`` →
    # ``@vue/language-server``).
    aliases = {
        "vue-language-server": "@vue/language-server",
        "astro-language-server": "@astrojs/language-server",
        "dockerfile-ls": "dockerfile-language-server-nodejs",
        "typescript": "typescript-language-server",
    }
    return aliases.get(server_id, server_id)


def _backend_warnings() -> list:
    """Return human-readable notes about LSP backend tools that are missing
    in a way that won't surface elsewhere.

    Some language servers ship as thin wrappers around an external CLI for
    actual diagnostics — they spawn cleanly but never emit any errors when
    the sidecar binary isn't on PATH.  bash-language-server / shellcheck
    is the load-bearing example.

    Returned strings are short, actionable, and include the install
    suggestion across common platforms.
    """
    import shutil as _shutil
    from agent.lsp.install import _existing_binary
    notes: list = []
    bash_installed = _existing_binary("bash-language-server") is not None
    if bash_installed and _shutil.which("shellcheck") is None:
        notes.append(
            "bash-language-server is installed but shellcheck is missing — "
            "diagnostics will be empty (apt: shellcheck, brew: shellcheck, "
            "scoop: shellcheck)."
        )
    return notes
