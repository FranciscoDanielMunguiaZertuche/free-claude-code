"""Track content-block state for native Anthropic SSE strings we emit to clients."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import suppress
from typing import Any

from core.anthropic.sse import SSEBuilder, format_sse_event
from core.anthropic.stream_contracts import SSEEvent, event_index, parse_sse_lines


class EmittedNativeSseTracker:
    """Parse emitted SSE frames so mid-stream errors can close blocks and pick a fresh index."""

    def __init__(self) -> None:
        self._buf = ""
        self._open_stack: list[int] = []
        self._block_types: dict[int, str] = {}
        self._tool_json_fragments: dict[int, list[str]] = {}
        self._max_index = -1
        self.message_id: str | None = None
        self.model: str = ""

    def feed(self, chunk: str) -> None:
        """Record SSE frames completed by ``chunk`` (handles splitting across reads)."""
        self._buf += chunk
        while True:
            sep = self._buf.find("\n\n")
            if sep < 0:
                break
            frame = self._buf[:sep]
            self._buf = self._buf[sep + 2 :]
            if not frame.strip():
                continue
            for event in parse_sse_lines(frame.splitlines()):
                self._observe(event)

    def _observe(self, event: SSEEvent) -> None:
        if event.event == "message_start":
            message = event.data.get("message")
            if isinstance(message, dict):
                mid = message.get("id")
                if isinstance(mid, str) and mid:
                    self.message_id = mid
                model = message.get("model")
                if isinstance(model, str) and model:
                    self.model = model
            return

        if event.event == "content_block_start":
            idx = event_index(event)
            self._max_index = max(self._max_index, idx)
            self._open_stack.append(idx)
            block = event.data.get("content_block", {})
            if isinstance(block, dict):
                self._block_types[idx] = str(block.get("type", ""))
            return

        if event.event == "content_block_delta":
            idx = event.data.get("index")
            if not isinstance(idx, int):
                return
            delta = event.data.get("delta", {})
            if (
                isinstance(delta, dict)
                and delta.get("type") == "input_json_delta"
                and self._block_types.get(idx) == "tool_use"
            ):
                partial = delta.get("partial_json", "")
                if partial:
                    self._tool_json_fragments.setdefault(idx, []).append(partial)
            return

        if event.event == "content_block_stop":
            idx = event.data.get("index")
            if isinstance(idx, int):
                if self._open_stack and self._open_stack[-1] == idx:
                    self._open_stack.pop()
                else:
                    with suppress(ValueError):
                        self._open_stack.remove(idx)
                self._block_types.pop(idx, None)
                self._tool_json_fragments.pop(idx, None)

    def next_content_index(self) -> int:
        """Next unused content block index based on emitted starts."""
        return self._max_index + 1

    def iter_close_unclosed_blocks(self) -> Iterator[str]:
        """Yield rescue deltas + ``content_block_stop`` for blocks started but not stopped.

        For tool_use blocks with invalid accumulated JSON, a rescue
        ``input_json_delta`` is emitted before ``content_block_stop`` so
        that downstream clients (Claude Code) don't fail with
        "tool call could not be parsed".
        """
        while self._open_stack:
            idx = self._open_stack.pop()
            if (
                self._block_types.get(idx) == "tool_use"
                and idx in self._tool_json_fragments
            ):
                concatenated = "".join(self._tool_json_fragments[idx])
                if concatenated.strip():
                    try:
                        json.loads(concatenated)
                    except json.JSONDecodeError, ValueError:
                        rescue = SSEBuilder._rescue_partial_json(concatenated)
                        yield format_sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": rescue,
                                },
                            },
                        )
            self._block_types.pop(idx, None)
            self._tool_json_fragments.pop(idx, None)
            yield format_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": idx},
            )

    def iter_midstream_error_tail(
        self,
        error_message: str,
        *,
        request: Any,
        input_tokens: int,
        log_raw_sse_events: bool,
    ) -> Iterator[str]:
        """Close dangling blocks, emit a text error block at a fresh index, then message tail."""
        mid = self.message_id or f"msg_{uuid.uuid4()}"
        model = self.model or (getattr(request, "model", "") or "")
        sse = SSEBuilder(
            mid,
            model,
            input_tokens,
            log_raw_events=log_raw_sse_events,
        )
        sse.blocks.next_index = self.next_content_index()
        yield from sse.emit_error(error_message)
        yield sse.message_delta("end_turn", 1)
        yield sse.message_stop()
