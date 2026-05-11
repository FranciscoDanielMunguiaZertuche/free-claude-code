"""Platform-neutral voice note helpers."""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .models import IncomingMessage


class PendingVoiceRegistry:
    """Track voice notes that are still waiting on transcription."""

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str], tuple[str, str]] = {}
        self._lock = asyncio.Lock()

    async def register(
        self, chat_id: str, voice_msg_id: str, status_msg_id: str
    ) -> None:
        async with self._lock:
            entry = (voice_msg_id, status_msg_id)
            self._pending[(chat_id, voice_msg_id)] = entry
            self._pending[(chat_id, status_msg_id)] = entry

    async def cancel(self, chat_id: str, reply_id: str) -> tuple[str, str] | None:
        async with self._lock:
            entry = self._pending.pop((chat_id, reply_id), None)
            if entry is None:
                return None
            voice_msg_id, status_msg_id = entry
            self._pending.pop((chat_id, voice_msg_id), None)
            self._pending.pop((chat_id, status_msg_id), None)
            return entry

    async def is_pending(self, chat_id: str, voice_msg_id: str) -> bool:
        async with self._lock:
            return (chat_id, voice_msg_id) in self._pending

    async def complete(
        self, chat_id: str, voice_msg_id: str, status_msg_id: str
    ) -> None:
        async with self._lock:
            self._pending.pop((chat_id, voice_msg_id), None)
            self._pending.pop((chat_id, status_msg_id), None)


class VoiceTranscriptionService:
    """Run configured transcription backends off the event loop."""

    def __init__(
        self,
        *,
        hf_token: str = "",
        nvidia_nim_api_key: str = "",
    ) -> None:
        self._hf_token = hf_token
        self._nvidia_nim_api_key = nvidia_nim_api_key

    async def transcribe(
        self,
        file_path: Path,
        mime_type: str,
        *,
        whisper_model: str,
        whisper_device: str,
    ) -> str:
        from .transcription import transcribe_audio

        return await asyncio.to_thread(
            transcribe_audio,
            file_path,
            mime_type,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            hf_token=self._hf_token,
            nvidia_nim_api_key=self._nvidia_nim_api_key,
        )


@dataclass(frozen=True, slots=True)
class VoiceNoteRequest:
    """Platform-neutral inputs for a single voice note transcription flow."""

    platform: str
    chat_id: str
    user_id: str
    message_id: str
    status_text: str
    status_reply_to: str
    mime_type: str
    temp_suffix: str
    download_to: Callable[[Path], Awaitable[None]]
    handler: Callable[[IncomingMessage], Awaitable[None]]
    status_parse_mode: str | None = None
    reply_to_message_id: str | None = None
    message_thread_id: str | None = None
    username: str | None = None
    raw_event: object | None = None


async def handle_voice_note_request(
    request: VoiceNoteRequest,
    *,
    queue_send_message: Callable[..., Awaitable[str | None]],
    queue_delete_message: Callable[..., Awaitable[None]],
    pending_voice: PendingVoiceRegistry,
    transcription: VoiceTranscriptionService,
    whisper_model: str,
    whisper_device: str,
    log_raw_messaging_content: bool,
) -> IncomingMessage | None:
    """Run common voice-note status, transcription, cancellation, and dispatch flow."""
    status_msg_id = await queue_send_message(
        request.chat_id,
        request.status_text,
        reply_to=request.status_reply_to,
        parse_mode=request.status_parse_mode,
        fire_and_forget=False,
        message_thread_id=request.message_thread_id,
    )
    status_id = str(status_msg_id)
    await pending_voice.register(request.chat_id, request.message_id, status_id)

    with tempfile.NamedTemporaryFile(suffix=request.temp_suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        await request.download_to(tmp_path)
        transcribed = await transcription.transcribe(
            tmp_path,
            request.mime_type,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
        )

        if not await pending_voice.is_pending(request.chat_id, request.message_id):
            await queue_delete_message(request.chat_id, status_id)
            return None

        await pending_voice.complete(request.chat_id, request.message_id, status_id)

        incoming = IncomingMessage(
            text=transcribed,
            chat_id=request.chat_id,
            user_id=request.user_id,
            message_id=request.message_id,
            platform=request.platform,
            reply_to_message_id=request.reply_to_message_id,
            message_thread_id=request.message_thread_id,
            username=request.username,
            raw_event=request.raw_event,
            status_message_id=status_id,
        )
        _log_voice_transcription(
            request.platform,
            request.chat_id,
            request.message_id,
            transcribed,
            log_raw_messaging_content=log_raw_messaging_content,
        )
        await request.handler(incoming)
        return incoming
    except Exception:
        await pending_voice.complete(request.chat_id, request.message_id, status_id)
        raise
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)


def _log_voice_transcription(
    platform: str,
    chat_id: str,
    message_id: str,
    transcribed: str,
    *,
    log_raw_messaging_content: bool,
) -> None:
    tag = platform.upper()
    if log_raw_messaging_content:
        preview = transcribed[:80] + "..." if len(transcribed) > 80 else transcribed
        logger.info(
            "{}_VOICE: chat_id={} message_id={} transcribed={!r}",
            tag,
            chat_id,
            message_id,
            preview,
        )
        return
    logger.info(
        "{}_VOICE: chat_id={} message_id={} transcribed_len={}",
        tag,
        chat_id,
        message_id,
        len(transcribed),
    )
