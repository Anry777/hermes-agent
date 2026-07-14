"""WebSocket transport coverage for the LSP client."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient
from agent.lsp.transports import LSPProtocolError, WebSocketLSPTransport

websockets = pytest.importorskip("websockets")


async def _ws_lsp_handler(ws):
    async for raw in ws:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        msg = json.loads(raw)
        if msg.get("method") == "initialize" and "id" in msg:
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": msg["id"],
                        "result": {
                            "capabilities": {
                                "textDocumentSync": 1,
                                "hoverProvider": True,
                                "definitionProvider": True,
                                "referencesProvider": True,
                                "documentSymbolProvider": True,
                            },
                            "serverInfo": {"name": "mock-websocket-lsp", "version": "0.1"},
                        },
                    },
                    separators=(",", ":"),
                )
            )
            continue
        if msg.get("method") == "initialized":
            continue
        if msg.get("method") == "textDocument/didOpen":
            params = msg.get("params") or {}
            td = params.get("textDocument") or {}
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "textDocument/publishDiagnostics",
                        "params": {
                            "uri": td.get("uri"),
                            "version": td.get("version", 0),
                            "diagnostics": [
                                {
                                    "range": {
                                        "start": {"line": 0, "character": 0},
                                        "end": {"line": 0, "character": 9},
                                    },
                                    "severity": 1,
                                    "code": "BSL001",
                                    "source": "mock-bsl-ls",
                                    "message": "synthetic websocket diagnostic",
                                }
                            ],
                        },
                    },
                    separators=(",", ":"),
                )
            )
            continue
        if msg.get("method") == "textDocument/didSave":
            continue
        if msg.get("method") == "shutdown" and "id" in msg:
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": None}))
            continue
        if msg.get("method") == "exit":
            await ws.close()
            return


@pytest.mark.asyncio
async def test_lsp_client_receives_diagnostics_over_websocket(tmp_path: Path):
    serve = websockets.serve
    async with serve(_ws_lsp_handler, "127.0.0.1", 0) as server:
        host, port = server.sockets[0].getsockname()[:2]
        url = f"ws://{host}:{port}/lsp"
        f = tmp_path / "module.bsl"
        f.write_text("Процедура Тест()\nКонецПроцедуры\n", encoding="utf-8")

        client = LSPClient(
            server_id="bsl-language-server",
            workspace_root=str(tmp_path),
            transport=WebSocketLSPTransport("bsl-language-server", url),
        )
        await client.start()
        try:
            assert client.is_running
            assert client.initialize_result is not None
            assert "capabilities" in client.initialize_result
            version = await client.open_file(str(f), language_id="bsl")
            await client.wait_for_diagnostics(str(f), version, mode="document")
            diags = client.diagnostics_for(str(f))
        finally:
            await client.shutdown()

    assert len(diags) == 1
    assert diags[0]["code"] == "BSL001"
    assert "websocket diagnostic" in diags[0]["message"]


@pytest.mark.asyncio
async def test_missing_websockets_dependency_fails_controlled(monkeypatch):
    from agent.lsp import transports

    def missing():
        raise ImportError("No module named websockets")

    monkeypatch.setattr(transports, "_import_websockets_connect", missing)
    transport = WebSocketLSPTransport("bsl-language-server", "ws://127.0.0.1:8025/lsp")

    with pytest.raises(LSPProtocolError, match="websockets package is required"):
        await asyncio.wait_for(transport.start(), timeout=1.0)
