"""Gateway media delivery requires explicit MEDIA: by default."""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class RecordingAdapter(BasePlatformAdapter):
    """Minimal adapter that records outbound text and native file sends."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent_texts: list[str] = []
        self.sent_documents: list[str] = []
        self.sent_videos: list[str] = []
        self.sent_image_batches: list[list[str]] = []
        self.sent_voices: list[str] = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        self.sent_texts.append(content)
        return SendResult(success=True, message_id=f"text-{len(self.sent_texts)}")

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption=None,
        file_name=None,
        reply_to=None,
        metadata=None,
        **kwargs,
    ) -> SendResult:
        self.sent_documents.append(file_path)
        return SendResult(success=True, message_id=f"doc-{len(self.sent_documents)}")

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption=None,
        reply_to=None,
        metadata=None,
        **kwargs,
    ) -> SendResult:
        self.sent_videos.append(video_path)
        return SendResult(success=True, message_id=f"video-{len(self.sent_videos)}")

    async def send_multiple_images(self, chat_id: str, images, metadata=None, human_delay=0.0) -> None:
        self.sent_image_batches.append([url for url, _alt in images])

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption=None,
        reply_to=None,
        metadata=None,
        **kwargs,
    ) -> SendResult:
        self.sent_voices.append(audio_path)
        return SendResult(success=True, message_id=f"voice-{len(self.sent_voices)}")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def _keep_typing(self, chat_id: str, interval: float = 2.0, metadata=None, stop_event=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"name": "Test chat", "type": "dm"}


@pytest.fixture(autouse=True)
def _gateway_media_env(monkeypatch):
    # Each test starts in the new safe default: bare local paths are not upload candidates.
    monkeypatch.delenv("HERMES_GATEWAY_AUTO_UPLOAD_LOCAL_PATHS", raising=False)
    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)
    monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)
    monkeypatch.delenv("HERMES_MEDIA_TRUST_RECENT_FILES", raising=False)
    monkeypatch.delenv("HERMES_MEDIA_TRUST_RECENT_SECONDS", raising=False)
    monkeypatch.setenv("HERMES_HUMAN_DELAY_MIN_MS", "0")
    monkeypatch.setenv("HERMES_HUMAN_DELAY_MAX_MS", "0")


def _event() -> MessageEvent:
    return MessageEvent(
        text="trigger",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", user_id="user-1"),
    )


async def _deliver_response(adapter: RecordingAdapter, response: str) -> None:
    async def handler(_event):
        return response

    adapter.set_message_handler(handler)
    await adapter._process_message_background(_event(), "telegram:chat-1")


@pytest.mark.asyncio
async def test_bare_local_path_is_text_by_default(tmp_path):
    report = tmp_path / "report.xlsx"
    report.write_bytes(b"xlsx")
    adapter = RecordingAdapter()

    await _deliver_response(adapter, f"Отчёт лежит тут: {report}")

    assert adapter.sent_documents == []
    assert adapter.sent_texts == [f"Отчёт лежит тут: {report}"]


@pytest.mark.asyncio
async def test_media_directive_still_sends_document_and_is_removed_from_text(tmp_path):
    report = tmp_path / "report.xlsx"
    report.write_bytes(b"xlsx")
    adapter = RecordingAdapter()

    await _deliver_response(adapter, f"Готово.\n\nMEDIA:{report}")

    assert adapter.sent_documents == [str(report)]
    assert adapter.sent_texts == ["Готово."]
    assert "MEDIA:" not in adapter.sent_texts[0]


@pytest.mark.asyncio
async def test_legacy_flag_restores_bare_local_path_upload(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_AUTO_UPLOAD_LOCAL_PATHS", "1")
    report = tmp_path / "report.xlsx"
    report.write_bytes(b"xlsx")
    adapter = RecordingAdapter()

    await _deliver_response(adapter, f"Отчёт: {report}")

    assert adapter.sent_documents == [str(report)]
    assert adapter.sent_texts == ["Отчёт:"]


@pytest.mark.asyncio
async def test_strict_still_blocks_legacy_bare_local_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_AUTO_UPLOAD_LOCAL_PATHS", "1")
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"secret")
    adapter = RecordingAdapter()

    await _deliver_response(adapter, f"Ключ: {secret}")

    assert adapter.sent_documents == []
    # Legacy extraction still strips candidates before validation, preserving old behavior
    # when the operator explicitly opts into bare-path auto-upload.
    assert adapter.sent_texts == ["Ключ:"]


@pytest.mark.asyncio
async def test_incident_style_profile_report_path_remains_text_by_default(tmp_path):
    report_dir = tmp_path / "root" / ".hermes" / "profiles" / "video_monitoring" / "reports"
    report_dir.mkdir(parents=True)
    report = report_dir / "video_monitoring_full_report_20260520_193404.xlsx"
    report.write_bytes(b"xlsx")
    adapter = RecordingAdapter()

    await _deliver_response(adapter, str(report))

    assert adapter.sent_documents == []
    assert adapter.sent_texts == [str(report)]


@pytest.mark.asyncio
async def test_post_stream_delivery_ignores_bare_paths_by_default(tmp_path):
    report = tmp_path / "post_stream.xlsx"
    report.write_bytes(b"xlsx")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingAdapter()

    await runner._deliver_media_from_response(f"Итог: {report}", _event(), adapter)

    assert adapter.sent_documents == []
    assert adapter.sent_image_batches == []


@pytest.mark.asyncio
async def test_post_stream_delivery_keeps_media_directive(tmp_path):
    report = tmp_path / "post_stream_media.xlsx"
    report.write_bytes(b"xlsx")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingAdapter()

    await runner._deliver_media_from_response(f"MEDIA:{report}", _event(), adapter)

    assert adapter.sent_documents == [str(report)]


@pytest.mark.asyncio
async def test_kanban_summary_bare_path_is_not_artifact_by_default(tmp_path):
    report = tmp_path / "summary.xlsx"
    report.write_bytes(b"xlsx")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingAdapter()

    await runner._deliver_kanban_artifacts(
        adapter=adapter,
        chat_id="chat-1",
        metadata={},
        event_payload={"summary": f"Готово: {report}"},
        task=None,
    )

    assert adapter.sent_documents == []


@pytest.mark.asyncio
async def test_kanban_structured_artifacts_remain_explicit_source(tmp_path):
    report = tmp_path / "artifact.xlsx"
    report.write_bytes(b"xlsx")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingAdapter()

    await runner._deliver_kanban_artifacts(
        adapter=adapter,
        chat_id="chat-1",
        metadata={},
        event_payload={"summary": "Готово", "artifacts": [str(report)]},
        task=None,
    )

    assert adapter.sent_documents == [str(report)]


def test_default_config_disables_bare_local_path_auto_upload():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["gateway"]["auto_upload_local_paths"] is False
