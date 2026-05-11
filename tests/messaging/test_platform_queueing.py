from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from messaging.models import IncomingMessage
from messaging.platforms.base import QueuedMessagingPlatform


class FakeQueuedPlatform(QueuedMessagingPlatform):
    name = "fake"

    def __init__(self, limiter: Any | None = None) -> None:
        self._limiter = limiter
        self.sent: list[tuple[str, str, str | None, str | None, str | None]] = []
        self.edited: list[tuple[str, str, str, str | None]] = []
        self.deleted: list[tuple[str, str]] = []
        self.bulk_deleted: list[tuple[str, tuple[str, ...]]] = []
        self._handler: Callable[[IncomingMessage], Awaitable[None]] | None = None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: str | None = None,
    ) -> str:
        self.sent.append((chat_id, text, reply_to, parse_mode, message_thread_id))
        return "sent-1"

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
    ) -> None:
        self.edited.append((chat_id, message_id, text, parse_mode))

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        self.deleted.append((chat_id, message_id))

    async def delete_messages(self, chat_id: str, message_ids: list[str]) -> None:
        self.bulk_deleted.append((chat_id, tuple(message_ids)))

    def on_message(
        self,
        handler: Callable[[IncomingMessage], Awaitable[None]],
    ) -> None:
        self._handler = handler


@pytest.mark.asyncio
async def test_queue_send_without_limiter_calls_send_message() -> None:
    platform = FakeQueuedPlatform()

    result = await platform.queue_send_message(
        "chat", "hello", reply_to="r1", parse_mode="md", message_thread_id="topic"
    )

    assert result == "sent-1"
    assert platform.sent == [("chat", "hello", "r1", "md", "topic")]


@pytest.mark.asyncio
async def test_queue_send_with_limiter_awaits_enqueue() -> None:
    limiter = AsyncMock()
    limiter.enqueue = AsyncMock(return_value="queued-id")
    platform = FakeQueuedPlatform(limiter)

    result = await platform.queue_send_message("chat", "hello", fire_and_forget=False)

    assert result == "queued-id"
    limiter.enqueue.assert_awaited_once()


@pytest.mark.asyncio
async def test_queue_edit_and_delete_use_stable_dedup_keys() -> None:
    limiter = AsyncMock()
    platform = FakeQueuedPlatform(limiter)

    await platform.queue_edit_message("chat", "m1", "updated", fire_and_forget=False)
    await platform.queue_delete_message("chat", "m1", fire_and_forget=False)
    await platform.queue_delete_messages("chat", ["m1", "m2"], fire_and_forget=False)

    assert limiter.enqueue.await_args_list[0].kwargs["dedup_key"] == "edit:chat:m1"
    assert limiter.enqueue.await_args_list[1].kwargs["dedup_key"] == "del:chat:m1"
    assert limiter.enqueue.await_args_list[2].kwargs["dedup_key"] == (
        f"del_bulk:chat:{hash(('m1', 'm2'))}"
    )


@pytest.mark.asyncio
async def test_queue_edit_fire_and_forget_uses_limiter_background_path() -> None:
    limiter = MagicMock()
    platform = FakeQueuedPlatform(limiter)

    await platform.queue_edit_message("chat", "m1", "updated")

    limiter.fire_and_forget.assert_called_once()
    assert limiter.fire_and_forget.call_args.kwargs["dedup_key"] == "edit:chat:m1"


def test_fire_and_forget_uses_create_task_for_coroutine() -> None:
    platform = FakeQueuedPlatform()

    async def _task() -> None:
        return None

    coro = _task()
    with patch("asyncio.create_task") as create_task:
        platform.fire_and_forget(coro)
        create_task.assert_called_once()
    coro.close()


def test_fire_and_forget_uses_ensure_future_for_future_like() -> None:
    platform = FakeQueuedPlatform()
    awaitable = MagicMock()

    with patch("asyncio.ensure_future") as ensure_future:
        platform.fire_and_forget(awaitable)

    ensure_future.assert_called_once_with(awaitable)
