"""Transport layer for Hermes LSP clients.

The LSP client speaks JSON-RPC envelopes.  How those envelopes move is a
transport concern: built-in servers normally use stdio Content-Length framing,
while externally managed servers can expose one JSON-RPC message per WebSocket
frame.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlparse

from agent.lsp.protocol import LSPProtocolError, encode_message, read_message

logger = logging.getLogger("agent.lsp.transports")


class LSPTransport(Protocol):
    """Minimal async JSON-RPC envelope transport for :class:`LSPClient`."""

    async def start(self) -> None: ...

    async def send(self, payload: dict) -> None: ...

    async def recv(self) -> Optional[dict]: ...

    async def close(self) -> None: ...

    @property
    def is_running(self) -> bool: ...

    def describe(self) -> str: ...


def validate_websocket_url(url: str) -> Optional[str]:
    """Return an error string when *url* is not a usable ws/wss URL."""
    parsed = urlparse(str(url or ""))
    if parsed.scheme not in {"ws", "wss"}:
        return "invalid websocket url: scheme must be ws or wss"
    if not parsed.netloc:
        return "invalid websocket url: missing host"
    return None


def _import_websockets_connect():
    try:
        import websockets  # type: ignore
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ImportError("websockets package is required for LSP websocket transport") from e
    return websockets.connect


def websocket_dependency_available() -> bool:
    try:
        _import_websockets_connect()
        return True
    except ImportError:
        return False


class StdioLSPTransport:
    """Content-Length framed LSP transport over a subprocess stdio pair."""

    def __init__(
        self,
        server_id: str,
        command: List[str],
        *,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        self.server_id = server_id
        self.command = list(command)
        self.env = env
        self.cwd = cwd
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def describe(self) -> str:
        return "stdio " + " ".join(self.command)

    async def start(self) -> None:
        if not self.command:
            raise LSPProtocolError("stdio LSP transport requires a command")
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.command[0],
                *self.command[1:],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.cwd,
            )
        except FileNotFoundError as e:
            raise LSPProtocolError(
                f"LSP server binary not found: {self.command[0]} ({e})"
            ) from e
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug("[%s] stderr: %s", self.server_id, text[:1000])
        except (asyncio.CancelledError, OSError):
            pass

    async def send(self, payload: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.is_closing():
            raise LSPProtocolError("cannot send: stdio stdin closed")
        try:
            proc.stdin.write(encode_message(payload))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise LSPProtocolError(f"stdio send failed: {e}") from e

    async def recv(self) -> Optional[dict]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        return await read_message(proc.stdout)

    async def close(self) -> None:
        if self._stderr_task is not None and not self._stderr_task.done():
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.2)
                return
            except asyncio.TimeoutError:
                pass
        if proc.returncode is None:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                        await proc.wait()
                    except ProcessLookupError:
                        pass
            except ProcessLookupError:
                pass


class WebSocketLSPTransport:
    """LSP transport where each JSON-RPC envelope is one WebSocket frame."""

    def __init__(self, server_id: str, url: str) -> None:
        self.server_id = server_id
        self.url = str(url or "")
        self._ws: Any = None

    @property
    def is_running(self) -> bool:
        return self._ws is not None

    def describe(self) -> str:
        return f"websocket {self.url}"

    async def start(self) -> None:
        url_error = validate_websocket_url(self.url)
        if url_error:
            raise LSPProtocolError(url_error)
        try:
            connect = _import_websockets_connect()
        except ImportError as e:
            raise LSPProtocolError("websockets package is required for LSP websocket transport") from e
        try:
            self._ws = await connect(self.url)
        except Exception as e:  # noqa: BLE001
            raise LSPProtocolError(f"websocket connect failed: {type(e).__name__}: {e}") from e

    async def send(self, payload: dict) -> None:
        if self._ws is None:
            raise LSPProtocolError("cannot send: websocket is not connected")
        try:
            await self._ws.send(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            raise LSPProtocolError(f"websocket send failed: {type(e).__name__}: {e}") from e

    async def recv(self) -> Optional[dict]:
        if self._ws is None:
            return None
        try:
            frame = await self._ws.recv()
        except Exception as e:  # noqa: BLE001
            raise LSPProtocolError(f"websocket recv failed: {type(e).__name__}: {e}") from e
        if frame is None:
            return None
        if isinstance(frame, bytes):
            try:
                frame = frame.decode("utf-8")
            except UnicodeDecodeError as e:
                raise LSPProtocolError(f"non-UTF-8 websocket LSP frame: {e}") from e
        if not isinstance(frame, str):
            raise LSPProtocolError(f"unsupported websocket LSP frame type: {type(frame).__name__}")
        try:
            msg = json.loads(frame)
        except json.JSONDecodeError as e:
            raise LSPProtocolError(f"invalid JSON websocket LSP frame: {e}") from e
        if not isinstance(msg, dict):
            raise LSPProtocolError("websocket LSP frame must decode to a JSON object")
        return msg

    async def close(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "LSPTransport",
    "StdioLSPTransport",
    "WebSocketLSPTransport",
    "LSPProtocolError",
    "validate_websocket_url",
    "websocket_dependency_available",
]
