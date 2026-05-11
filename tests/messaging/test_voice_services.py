from pathlib import Path
from unittest.mock import patch

import pytest

from messaging.voice import (
    PendingVoiceRegistry,
    VoiceNoteRequest,
    VoiceTranscriptionService,
    handle_voice_note_request,
)


@pytest.mark.asyncio
async def test_pending_voice_registry_tracks_voice_and_status_ids():
    registry = PendingVoiceRegistry()

    await registry.register("chat", "voice-1", "status-1")

    assert await registry.is_pending("chat", "voice-1") is True
    assert await registry.cancel("chat", "status-1") == ("voice-1", "status-1")
    assert await registry.is_pending("chat", "voice-1") is False


@pytest.mark.asyncio
async def test_pending_voice_registry_complete_removes_entries():
    registry = PendingVoiceRegistry()

    await registry.register("chat", "voice-1", "status-1")
    await registry.complete("chat", "voice-1", "status-1")

    assert await registry.cancel("chat", "voice-1") is None


@pytest.mark.asyncio
async def test_voice_transcription_service_runs_backend():
    service = VoiceTranscriptionService()

    with patch("messaging.transcription.transcribe_audio", return_value="hello"):
        text = await service.transcribe(
            Path("audio.ogg"),
            "audio/ogg",
            whisper_model="base",
            whisper_device="cpu",
        )

    assert text == "hello"


@pytest.mark.asyncio
async def test_voice_note_flow_dispatches_transcribed_incoming_message():
    registry = PendingVoiceRegistry()
    service = VoiceTranscriptionService()
    sent = []
    handled = []

    async def queue_send_message(*args, **kwargs):
        sent.append((args, kwargs))
        return "status-1"

    async def queue_delete_message(*args, **kwargs):
        raise AssertionError("status should not be deleted on success")

    async def download_to(path: Path):
        path.write_bytes(b"audio")

    async def handler(incoming):
        handled.append(incoming)

    request = VoiceNoteRequest(
        platform="telegram",
        chat_id="chat",
        user_id="user",
        message_id="voice-1",
        status_text="Transcribing",
        status_reply_to="voice-1",
        status_parse_mode="MarkdownV2",
        mime_type="audio/ogg",
        temp_suffix=".ogg",
        download_to=download_to,
        handler=handler,
        reply_to_message_id="parent",
        message_thread_id="topic",
        raw_event={"raw": True},
    )

    with patch("messaging.transcription.transcribe_audio", return_value="hello"):
        incoming = await handle_voice_note_request(
            request,
            queue_send_message=queue_send_message,
            queue_delete_message=queue_delete_message,
            pending_voice=registry,
            transcription=service,
            whisper_model="base",
            whisper_device="cpu",
            log_raw_messaging_content=False,
        )

    assert incoming is handled[0]
    assert incoming.text == "hello"
    assert incoming.status_message_id == "status-1"
    assert incoming.message_thread_id == "topic"
    assert sent[0][1]["parse_mode"] == "MarkdownV2"
    assert await registry.cancel("chat", "voice-1") is None


@pytest.mark.asyncio
async def test_voice_note_flow_cancel_deletes_status_without_dispatch():
    registry = PendingVoiceRegistry()
    service = VoiceTranscriptionService()
    deleted = []
    handled = []

    async def queue_send_message(*_args, **_kwargs):
        return "status-1"

    async def queue_delete_message(chat_id, message_id, **_kwargs):
        deleted.append((chat_id, message_id))

    async def download_to(path: Path):
        path.write_bytes(b"audio")
        await registry.cancel("chat", "voice-1")

    async def handler(incoming):
        handled.append(incoming)

    request = VoiceNoteRequest(
        platform="discord",
        chat_id="chat",
        user_id="user",
        message_id="voice-1",
        status_text="Transcribing",
        status_reply_to="voice-1",
        mime_type="audio/ogg",
        temp_suffix=".ogg",
        download_to=download_to,
        handler=handler,
    )

    with patch("messaging.transcription.transcribe_audio", return_value="hello"):
        incoming = await handle_voice_note_request(
            request,
            queue_send_message=queue_send_message,
            queue_delete_message=queue_delete_message,
            pending_voice=registry,
            transcription=service,
            whisper_model="base",
            whisper_device="cpu",
            log_raw_messaging_content=False,
        )

    assert incoming is None
    assert handled == []
    assert deleted == [("chat", "status-1")]


@pytest.mark.asyncio
async def test_voice_note_flow_failure_clears_pending_registry():
    registry = PendingVoiceRegistry()
    service = VoiceTranscriptionService()

    async def queue_send_message(*_args, **_kwargs):
        return "status-1"

    async def queue_delete_message(*_args, **_kwargs):
        return None

    async def download_to(path: Path):
        path.write_bytes(b"audio")

    async def handler(_incoming):
        raise AssertionError("handler should not run")

    request = VoiceNoteRequest(
        platform="telegram",
        chat_id="chat",
        user_id="user",
        message_id="voice-1",
        status_text="Transcribing",
        status_reply_to="voice-1",
        mime_type="audio/ogg",
        temp_suffix=".ogg",
        download_to=download_to,
        handler=handler,
    )

    with (
        patch(
            "messaging.transcription.transcribe_audio", side_effect=ValueError("bad")
        ),
        pytest.raises(ValueError, match="bad"),
    ):
        await handle_voice_note_request(
            request,
            queue_send_message=queue_send_message,
            queue_delete_message=queue_delete_message,
            pending_voice=registry,
            transcription=service,
            whisper_model="base",
            whisper_device="cpu",
            log_raw_messaging_content=False,
        )

    assert await registry.cancel("chat", "voice-1") is None
    assert await registry.cancel("chat", "status-1") is None
