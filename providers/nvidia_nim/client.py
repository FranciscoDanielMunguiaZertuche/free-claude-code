"""NVIDIA NIM provider implementation."""

import asyncio
import json
import threading
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import openai
from loguru import logger

from config.nim import NimSettings
from core.anthropic import (
    ContentType,
    HeuristicToolParser,
    SSEBuilder,
    ThinkTagParser,
    append_request_id,
    map_stop_reason,
)
from core.anthropic.sse import ToolCallState
from providers.base import ProviderConfig
from providers.defaults import NVIDIA_NIM_DEFAULT_BASE
from providers.error_mapping import (
    map_error,
    user_visible_message_for_mapped_provider_error,
)
from providers.openai_compat import OpenAIChatTransport, _iter_heuristic_tool_use_sse

from .request import (
    body_without_nim_tool_argument_aliases,
    build_request_body,
    clone_body_without_all_thinking,
    clone_body_without_chat_template,
    clone_body_without_reasoning_budget,
    clone_body_without_reasoning_content,
    nim_tool_argument_aliases_from_body,
)


def _convert_nonstream_to_sse(
    response: Any,
    sse: SSEBuilder,
    thinking_enabled: bool,
) -> list[str]:
    """Convert a non-streaming OpenAI chat completion to Anthropic SSE events.

    NVIDIA's streaming with tools drops id/type/function.name from tool_call
    chunks. Non-streaming returns complete tool_call objects,
    so we send ``stream=False`` and convert the full JSON response here.
    """
    events: list[str] = []
    choice = response.choices[0]
    message = choice.message
    finish_reason = choice.finish_reason

    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        heuristic_parser = HeuristicToolParser()
        filtered_text, detected_tools = heuristic_parser.feed(reasoning)
        if filtered_text:
            events.extend(sse.ensure_text_block())
            events.append(sse.emit_text_delta(filtered_text))
        for tool_use in detected_tools:
            events.extend(_iter_heuristic_tool_use_sse(sse, tool_use))
        for tool_use in heuristic_parser.flush():
            events.extend(_iter_heuristic_tool_use_sse(sse, tool_use))

        has_detected_tools = any(s.started for s in sse.blocks.tool_states.values())
        if not has_detected_tools:
            _xml_remaining, xml_tools = HeuristicToolParser.extract_xml_tool_calls(
                reasoning
            )
            for tool_use in xml_tools:
                events.extend(_iter_heuristic_tool_use_sse(sse, tool_use))

    content = message.content
    if content:
        think_parser = ThinkTagParser()
        heuristic_parser_content = HeuristicToolParser()
        for part in think_parser.feed(content):
            if part.type == ContentType.THINKING:
                if not thinking_enabled:
                    continue
                events.extend(sse.ensure_text_block())
                events.append(sse.emit_text_delta(part.content))
            else:
                filtered_text, detected_tools = heuristic_parser_content.feed(
                    part.content
                )
                if filtered_text:
                    events.extend(sse.ensure_text_block())
                    events.append(sse.emit_text_delta(filtered_text))
                for tool_use in detected_tools:
                    events.extend(_iter_heuristic_tool_use_sse(sse, tool_use))
        remaining = think_parser.flush()
        if remaining and (
            remaining.type == ContentType.TEXT
            or (remaining.type == ContentType.THINKING and thinking_enabled)
        ):
            events.extend(sse.ensure_text_block())
            events.append(sse.emit_text_delta(remaining.content))
        for tool_use in heuristic_parser_content.flush():
            events.extend(_iter_heuristic_tool_use_sse(sse, tool_use))

    tool_calls = message.tool_calls
    has_started_tool = any(s.started for s in sse.blocks.tool_states.values())
    if tool_calls:
        events.extend(sse.close_content_blocks())
        for i, tc in enumerate(tool_calls):
            tool_id = str(tc.id) if tc.id else f"tool_{uuid.uuid4()}"
            tool_name = (tc.function.name or "tool_call").strip()
            block_idx = sse.blocks.allocate_index()
            sse.blocks.tool_states[i] = ToolCallState(
                block_index=block_idx,
                tool_id=tool_id,
                name=tool_name,
                started=True,
            )
            events.append(
                sse.content_block_start(
                    block_idx, "tool_use", id=tool_id, name=tool_name
                )
            )
            args = tc.function.arguments or "{}"
            try:
                json.loads(args)
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "NIM_NONSTREAM: tool_call arguments not valid JSON (id={} len={}), falling back to {{}}",
                    tool_id,
                    len(args),
                )
                args = "{}"
            events.append(sse.emit_tool_delta(i, args))
        has_started_tool = any(s.started for s in sse.blocks.tool_states.values())
    has_content_blocks = sse.blocks.text_index != -1 or has_started_tool
    if not has_content_blocks:
        events.extend(sse.ensure_text_block())
        events.append(sse.emit_text_delta(" "))

    events.extend(sse.close_all_blocks())

    if has_started_tool and finish_reason != "tool_calls":
        finish_reason = "tool_calls"

    usage = getattr(response, "usage", None)
    output_tokens = (
        usage.completion_tokens
        if usage and hasattr(usage, "completion_tokens")
        else sse.estimate_output_tokens()
    )
    events.append(sse.message_delta(map_stop_reason(finish_reason), output_tokens))
    events.append(sse.message_stop())
    return events


class NvidiaNimProvider(OpenAIChatTransport):
    """NVIDIA NIM provider using official OpenAI client.

    Supports round-robin load balancing across multiple API keys, with
    automatic failover on 429 rate limit and read timeouts. Each request
    starts from the next key in rotation; if that key fails, remaining
    keys are tried in rotation until all are exhausted."""

    _reasoning_content_is_text: bool = True
    """NIM's ``reasoning_content`` contains the actual response text,
    not separate chain-of-thought. Emit it as regular text."""

    _RETRYABLE_EXC: tuple[type[Exception], ...] = (
        openai.RateLimitError,
        openai.APITimeoutError,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
        httpx.ConnectError,
        httpx.ConnectTimeout,
        ConnectionError,
    )

    _NETWORK_DOWN_EXC: tuple[type[Exception], ...] = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        ConnectionError,
    )  # Exceptions indicating network is unreachable — retrying other keys is pointless.

    @staticmethod
    def _log_nim_error(exc: Exception, tag: str, req_tag: str) -> None:
        """Log upstream NIM error with status code and response body."""
        status_code = getattr(exc, "status_code", None)
        error_body = getattr(exc, "body", None)
        if error_body is not None:
            logger.error(
                "{}_ERROR:{} status={} body={}",
                tag,
                req_tag,
                status_code,
                json.dumps(error_body, default=str),
            )
        else:
            logger.error(
                "{}_ERROR:{} exc_type={} status={}",
                tag,
                req_tag,
                type(exc).__name__,
                status_code,
            )

    def __init__(
        self,
        config: ProviderConfig,
        *,
        nim_settings: NimSettings,
        fallback_api_keys: list[str] | None = None,
    ):
        super().__init__(
            config,
            provider_name="NIM",
            base_url=config.base_url or NVIDIA_NIM_DEFAULT_BASE,
            api_key=config.api_key,
        )
        self._nim_settings = nim_settings
        self._fallback_clients: list[openai.AsyncOpenAI] = []
        self._rr_index: int = 0
        self._rr_lock = threading.Lock()
        if fallback_api_keys:
            for key in fallback_api_keys:
                _http_client = None
                if config.proxy:
                    _http_client = httpx.AsyncClient(
                        proxy=config.proxy,
                        timeout=httpx.Timeout(
                            config.http_read_timeout,
                            connect=config.http_connect_timeout,
                            read=config.http_read_timeout,
                            write=config.http_write_timeout,
                        ),
                    )
                self._fallback_clients.append(
                    openai.AsyncOpenAI(
                        api_key=key,
                        base_url=self._base_url,
                        max_retries=0,
                        timeout=httpx.Timeout(
                            config.http_read_timeout,
                            connect=config.http_connect_timeout,
                            read=config.http_read_timeout,
                            write=config.http_write_timeout,
                        ),
                        http_client=_http_client,
                    )
                )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(
            request,
            self._nim_settings,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Strip private request metadata before calling NVIDIA NIM."""
        return body_without_nim_tool_argument_aliases(body)

    def _tool_argument_aliases(self, body: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Return NIM tool argument aliases captured while building this request."""
        return nim_tool_argument_aliases_from_body(body)

    async def _create_stream(self, body: dict) -> tuple[Any, dict]:
        """Create a streaming chat completion with timeout retry across keys.

        NIM upstream stalls are transient — the server sometimes never sends
        response headers within the read timeout, but a retry typically
        succeeds instantly. We retry once on the same key after a short delay,
        then advance through remaining keys in round-robin order.

        ``self._client`` is already set to the correct key by the caller
        (``_streaming_path_with_fallback``), so the initial attempt uses it
        directly. On failure we rotate through ``_fallback_clients``.
        """
        create_body = self._prepare_create_body(body)

        async def _try_create(client: openai.AsyncOpenAI) -> Any:
            return await self._global_rate_limiter.execute_with_retry(
                client.chat.completions.create, **create_body, stream=True
            )

        # 1. First attempt on the already-selected client
        hit_timeout = False
        last_exc: Exception | None = None
        try:
            stream = await _try_create(self._client)
            return stream, body
        except (openai.APITimeoutError, httpx.ReadTimeout) as e:
            hit_timeout = True
            last_exc = e
            logger.warning(
                "NIM_CREATE_STREAM: {} on primary key, retrying same key after 5s",
                type(e).__name__,
            )
        except self._RETRYABLE_EXC as e:
            last_exc = e
            logger.warning(
                "NIM_CREATE_STREAM: {} on primary key, advancing to fallbacks",
                type(e).__name__,
            )
        except openai.BadRequestError as e:
            retry_body = self._get_retry_request_body(e, body)
            if retry_body is None:
                raise
            create_retry_body = self._prepare_create_body(retry_body)
            try:
                stream = await self._global_rate_limiter.execute_with_retry(
                    self._client.chat.completions.create,
                    **create_retry_body,
                    stream=True,
                )
                return stream, retry_body
            except self._RETRYABLE_EXC:
                pass

        # 2. Same-key timeout retry — only when the first attempt timed out.
        # NIM stalls are transient; a retry typically succeeds instantly.
        # Rate limits (429) skip this and go straight to fallback keys.
        if hit_timeout:
            try:
                await asyncio.sleep(5)
                stream = await _try_create(self._client)
                return stream, body
            except (openai.APITimeoutError, httpx.ReadTimeout) as e:
                last_exc = e
                logger.warning(
                    "NIM_CREATE_STREAM: timeout on same-key retry, trying fallbacks"
                )
            except self._RETRYABLE_EXC as e:
                last_exc = e
                logger.warning(
                    "NIM_CREATE_STREAM: {} on same-key retry, trying fallbacks",
                    type(e).__name__,
                )

        # 3. Fallback keys
        for fb_idx, fb_client in enumerate(self._fallback_clients):
            logger.warning(
                "NIM_CREATE_STREAM: trying fallback key #{}/{}",
                fb_idx + 1,
                len(self._fallback_clients),
            )
            try:
                stream = await _try_create(fb_client)
                return stream, body
            except (openai.APITimeoutError, httpx.ReadTimeout) as e:
                last_exc = e
                try:
                    await asyncio.sleep(5)
                    stream = await _try_create(fb_client)
                    return stream, body
                except self._RETRYABLE_EXC as retry_exc:
                    last_exc = retry_exc
            except self._RETRYABLE_EXC as e:
                last_exc = e
            except openai.BadRequestError as e:
                retry_body = self._get_retry_request_body(e, body)
                if retry_body is None:
                    raise
                create_retry_body = self._prepare_create_body(retry_body)
                try:
                    stream = await self._global_rate_limiter.execute_with_retry(
                        fb_client.chat.completions.create,
                        **create_retry_body,
                        stream=True,
                    )
                    return stream, retry_body
                except self._RETRYABLE_EXC as retry_exc:
                    last_exc = retry_exc

        if last_exc is not None:
            raise last_exc
        raise openai.APITimeoutError(request=httpx.Request("POST", self._base_url))

    def _next_client(self) -> tuple[openai.AsyncOpenAI, int]:
        """Return the next client in round-robin rotation and its index.

        Uses a threading lock so concurrent async tasks get distinct keys.
        """
        all_clients = [self._client, *self._fallback_clients]
        with self._rr_lock:
            idx = self._rr_index % len(all_clients)
            self._rr_index += 1
        return all_clients[idx], idx

    def _rr_client_order(self) -> list[openai.AsyncOpenAI]:
        """Return all clients in round-robin order starting from the next key."""
        all_clients = [self._client, *self._fallback_clients]
        _, start_idx = self._next_client()
        return [
            all_clients[(start_idx + i) % len(all_clients)]
            for i in range(len(all_clients))
        ]

    async def _stream_response_impl(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
        *,
        thinking_enabled: bool | None,
    ) -> AsyncIterator[str]:
        """NIM-optimized streaming with adaptive tool-call strategy.

        For models where ``reasoning_content`` contains the actual response text
        (``_reasoning_content_is_text=True``), NIM's non-streaming endpoint puts
        tool calls in ``reasoning_content`` as XML text (e.g.
        ``<Read><filePath>/tmp/x</filePath></Read>``) instead of structured
        ``tool_calls``. Streaming, however, returns proper structured tool_call
        chunks. So when ``_reasoning_content_is_text`` is True and tools are
        present, we use streaming directly.

        For other NIM models where streaming drops id/type/function.name from
        tool_call chunks (a known NVIDIA platform bug), we send ``stream=False``
        for tool-bearing requests and convert the complete JSON response to
        Anthropic SSE events. The ``HeuristicToolParser`` catches ``● <function>``
        patterns and the new ``extract_xml_tool_calls`` catches NIM XML-style
        tool calls in ``reasoning_content``.

        Text-only and thinking-only requests always use streaming for lower TTFB.

        When non-streaming fails (e.g. 429 rate limit, timeout), we fall back
        to streaming with tool-call fixup (the parent's ``_process_tool_call``
        generates missing IDs, buffers names, etc.).
        """
        body = self._build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
        thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)
        has_tools = bool(body.get("tools"))

        if not has_tools:
            async for event in self._streaming_path_with_fallback(
                request, input_tokens, request_id, thinking_enabled=thinking_enabled
            ):
                yield event
            return

        if self._reasoning_content_is_text:
            async for event in self._streaming_path_with_fallback(
                request, input_tokens, request_id, thinking_enabled=thinking_enabled
            ):
                yield event
            return

        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(
            message_id,
            request.model,
            input_tokens,
            log_raw_events=self._config.log_raw_sse_events,
        )
        req_tag = f" request_id={request_id}" if request_id else ""
        logger.info(
            "NIM_NONSTREAM:{} model={} msgs={} tools={} (tools present, using stream=false)",
            req_tag,
            body.get("model"),
            len(body.get("messages", [])),
            len(body.get("tools", [])),
        )

        rr_order = self._rr_client_order()
        async with self._global_rate_limiter.concurrency_slot():
            nonstream_exc: Exception | None = None
            is_rate_limit = False
            for client_idx, active_client in enumerate(rr_order):
                if client_idx > 0:
                    logger.warning(
                        "NIM_NONSTREAM:{} trying fallback key #{}/{}",
                        req_tag,
                        client_idx,
                        len(self._fallback_clients),
                    )
                try:
                    response = await self._global_rate_limiter.execute_with_retry(
                        active_client.chat.completions.create,
                        **body,
                        stream=False,
                        max_retries=6,
                        max_delay=120.0,
                    )
                    yield sse.message_start()
                    for event in _convert_nonstream_to_sse(
                        response, sse, thinking_enabled
                    ):
                        yield event
                    return
                except openai.RateLimitError as e:
                    nonstream_exc = e
                    is_rate_limit = True
                    logger.warning(
                        "NIM_NONSTREAM:{} rate limit on key #{}/{}, {}",
                        req_tag,
                        client_idx + 1,
                        len(rr_order),
                        type(e).__name__,
                    )
                except (openai.APITimeoutError, httpx.ReadTimeout) as e:
                    nonstream_exc = e
                    logger.warning(
                        "NIM_NONSTREAM:{} timeout on key #{}/{}, {}",
                        req_tag,
                        client_idx + 1,
                        len(rr_order),
                        type(e).__name__,
                    )
                except openai.BadRequestError as e:
                    retry_body = self._get_retry_request_body(e, body)
                    if retry_body is not None:
                        logger.warning(
                            "NIM_NONSTREAM:{} retrying with modified body after 400",
                            req_tag,
                        )
                        try:
                            response = (
                                await self._global_rate_limiter.execute_with_retry(
                                    active_client.chat.completions.create,
                                    **retry_body,
                                    stream=False,
                                )
                            )
                            yield sse.message_start()
                            for event in _convert_nonstream_to_sse(
                                response, sse, thinking_enabled
                            ):
                                yield event
                            return
                        except openai.BadRequestError as retry_exc:
                            retry_body_2 = self._get_retry_request_body(
                                retry_exc, retry_body
                            )
                            if retry_body_2 is not None:
                                logger.warning(
                                    "NIM_NONSTREAM:{} second retry after 400 on modified body",
                                    req_tag,
                                )
                                try:
                                    response = await self._global_rate_limiter.execute_with_retry(
                                        active_client.chat.completions.create,
                                        **retry_body_2,
                                        stream=False,
                                    )
                                    yield sse.message_start()
                                    for event in _convert_nonstream_to_sse(
                                        response, sse, thinking_enabled
                                    ):
                                        yield event
                                    return
                                except Exception as retry_exc_2:
                                    nonstream_exc = retry_exc_2
                                    body = retry_body_2
                                    break
                            nonstream_exc = retry_exc
                            body = retry_body
                            break
                        except Exception as retry_exc:
                            nonstream_exc = retry_exc
                            body = retry_body
                            break
                    else:
                        nonstream_exc = e
                        break
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    nonstream_exc = e

        if nonstream_exc is not None and not is_rate_limit:
            logger.warning(
                "NIM_NONSTREAM:{} non-streaming failed ({}), retrying with streaming",
                req_tag,
                type(nonstream_exc).__name__,
            )
            for client_idx, active_client in enumerate(rr_order):
                if client_idx > 0:
                    logger.warning(
                        "NIM_NONSTREAM:{} streaming fallback on key #{}/{}",
                        req_tag,
                        client_idx,
                        len(self._fallback_clients),
                    )
                try:
                    yield sse.message_start()
                    async for event in self._streaming_fallback_with_client(
                        body, sse, thinking_enabled, req_tag, active_client
                    ):
                        yield event
                    return
                except self._RETRYABLE_EXC:
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception:
                    continue
            logger.error(
                "NIM_NONSTREAM:{} streaming fallback also failed on all keys",
                req_tag,
            )

        if nonstream_exc is not None:
            self._log_nim_error(nonstream_exc, "NIM_NONSTREAM", req_tag)
            yield sse.message_start()
            for event in sse.close_incomplete_tool_blocks():
                yield event
            for event in sse.close_all_blocks():
                yield event
            mapped_e = map_error(nonstream_exc, rate_limiter=self._global_rate_limiter)
            base_message = user_visible_message_for_mapped_provider_error(
                mapped_e,
                provider_name="NIM",
                read_timeout_s=self._config.http_read_timeout,
            )
            error_message = append_request_id(base_message, request_id)
            if sse.blocks.has_emitted_tool_block():
                yield sse.emit_top_level_error(error_message)
            else:
                for event in sse.emit_error(error_message):
                    yield event
            yield sse.message_delta("end_turn", 1)
            yield sse.message_stop()
            return

    async def _streaming_fallback(
        self,
        body: dict,
        sse: SSEBuilder,
        thinking_enabled: bool,
        req_tag: str,
        client: openai.AsyncOpenAI | None = None,
    ) -> AsyncIterator[str]:
        """Retry a tool-bearing request with streaming after non-streaming fails.

        NVIDIA's streaming with tools drops id/type/function.name from tool_call
        chunks. The parent ``_process_tool_call`` patches these up (generates
        missing IDs, buffers pre-start args, backfills names from request), so
        streaming produces usable results despite the broken chunks.
        """
        think_parser = ThinkTagParser()
        heuristic_parser = HeuristicToolParser()
        finish_reason = None
        usage_info = None

        active_client = client or self._client
        stream = await self._global_rate_limiter.execute_with_retry(
            active_client.chat.completions.create, **body, stream=True
        )

        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage_info = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is None:
                continue
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                filtered_text, detected_tools = heuristic_parser.feed(reasoning)
                if filtered_text:
                    for event in sse.ensure_text_block():
                        yield event
                    yield sse.emit_text_delta(filtered_text)
                for tool_use in detected_tools:
                    for event in _iter_heuristic_tool_use_sse(sse, tool_use):
                        yield event

            if delta.content:
                for part in think_parser.feed(delta.content):
                    if part.type == ContentType.THINKING:
                        if not thinking_enabled:
                            continue
                        for event in sse.ensure_text_block():
                            yield event
                        yield sse.emit_text_delta(part.content)
                    else:
                        filtered_text, detected_tools = heuristic_parser.feed(
                            part.content
                        )
                        if filtered_text:
                            for event in sse.ensure_text_block():
                                yield event
                            yield sse.emit_text_delta(filtered_text)
                        for tool_use in detected_tools:
                            for event in _iter_heuristic_tool_use_sse(sse, tool_use):
                                yield event

            if delta.tool_calls:
                for event in sse.close_content_blocks():
                    yield event
                for tc in delta.tool_calls:
                    tc_info = {
                        "index": tc.index,
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for event in self._process_tool_call(tc_info, sse):
                        yield event

        remaining = think_parser.flush()
        if remaining and (
            remaining.type == ContentType.TEXT
            or (remaining.type == ContentType.THINKING and thinking_enabled)
        ):
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(remaining.content)
        for tool_use in heuristic_parser.flush():
            for event in _iter_heuristic_tool_use_sse(sse, tool_use):
                yield event

        has_started_tool = any(s.started for s in sse.blocks.tool_states.values())
        has_content_blocks = sse.blocks.text_index != -1 or has_started_tool
        if not has_content_blocks:
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(" ")

        for event in self._flush_task_arg_buffers(sse):
            yield event
        for event in sse.close_incomplete_tool_blocks():
            yield event
        for event in sse.close_all_blocks():
            yield event

        output_tokens = (
            usage_info.completion_tokens
            if usage_info and hasattr(usage_info, "completion_tokens")
            else sse.estimate_output_tokens()
        )
        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()

    async def _streaming_fallback_with_client(
        self,
        body: dict,
        sse: SSEBuilder,
        thinking_enabled: bool,
        req_tag: str,
        client: openai.AsyncOpenAI,
    ) -> AsyncIterator[str]:
        """Streaming fallback using a specific client (for key rotation)."""
        async for event in self._streaming_fallback(
            body, sse, thinking_enabled, req_tag, client=client
        ):
            yield event

    async def _streaming_path_with_fallback(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
        *,
        thinking_enabled: bool | None,
    ) -> AsyncIterator[str]:
        """Stream with round-robin key selection and automatic fallback on transport errors.

        Timeout retry at the connection level is handled by ``_create_stream``.
        This method covers mid-stream transport errors (e.g. ``RemoteProtocolError``
        from NIM abruptly closing the connection). The parent's ``_stream_response_impl``
        catches these and emits error SSE events — we detect that and retry on the
        next key, but only when no content has been emitted yet (retrying after
        partial content would produce duplicate output).
        """
        req_tag = f" request_id={request_id}" if request_id else ""
        rr_order = self._rr_client_order()
        for attempt_idx, client in enumerate(rr_order):
            if attempt_idx > 0:
                logger.warning(
                    "NIM_STREAM:{} trying key #{}/{} after failure",
                    req_tag,
                    attempt_idx + 1,
                    len(rr_order),
                )
            _original = self._client
            try:
                self._client = client
                got_content = False
                pending_events: list[str] = []
                async for event in self._streaming_path(
                    request,
                    input_tokens,
                    request_id,
                    thinking_enabled=thinking_enabled,
                ):
                    is_error = (
                        '"type":"error"' in event or '"type": "error"' in event
                    ) and (
                        "peer closed" in event.lower()
                        or "incomplete chunked" in event.lower()
                        or "timed out" in event.lower()
                        or "connection" in event.lower()
                    )
                    is_content = (
                        '"content_block_start"' in event or '"text_delta"' in event
                    )
                    if is_content:
                        got_content = True
                    if is_error and not got_content:
                        logger.warning(
                            "NIM_STREAM:{} early transport error (no content yet), "
                            "raising for key rotation",
                            req_tag,
                        )
                        raise httpx.RemoteProtocolError(
                            "peer closed connection without sending complete message body"
                        )
                    if not got_content:
                        pending_events.append(event)
                    else:
                        for pending in pending_events:
                            yield pending
                        pending_events.clear()
                        yield event
                return
            except self._RETRYABLE_EXC:
                logger.warning(
                    "NIM_STREAM:{} transport error on key #{}/{}, advancing",
                    req_tag,
                    attempt_idx + 1,
                    len(rr_order),
                )
                continue
            finally:
                self._client = _original

    async def _streaming_path(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
        *,
        thinking_enabled: bool | None,
    ) -> AsyncIterator[str]:
        """Fall back to the parent streaming implementation."""
        async for event in super()._stream_response_impl(
            request, input_tokens, request_id, thinking_enabled=thinking_enabled
        ):
            yield event

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Retry with a downgraded body when NIM rejects a known field.

        Tries keyword-specific stripping first (reasoning_budget, chat_template,
        reasoning_content). If none match but the error hints at thinking/reasoning
        params, falls back to stripping ALL thinking-related extra_body fields.
        """
        status_code = getattr(error, "status_code", None)
        if not isinstance(error, openai.BadRequestError) and status_code != 400:
            return None

        error_text = str(error)
        error_body = getattr(error, "body", None)
        if error_body is not None:
            error_text = f"{error_text} {json.dumps(error_body, default=str)}"
        error_text_lower = error_text.lower()

        logger.warning(
            "NIM_NONSTREAM: 400 error body={}",
            json.dumps(error_body, default=str),
        )

        if "input_tokens" in error_text_lower:
            import re

            context_match = re.search(r"context length is only (\d+)", error_text)
            input_match = re.search(r"passed (\d+) input tokens", error_text)
            if context_match and input_match:
                context_len = int(context_match.group(1))
                input_tokens_count = int(input_match.group(1))
                current_max_tokens = body.get("max_tokens", 0) or 0
                excess = (input_tokens_count + current_max_tokens) - context_len
                if excess > 0:
                    retry_body = clone_body_without_all_thinking(body)
                    if retry_body is not None:
                        retry_body["max_tokens"] = max(
                            context_len - input_tokens_count - 2048, 1
                        )
                        logger.warning(
                            "NIM_NONSTREAM: context overflow (input={} + max_tokens={} > context={}), "
                            "stripped all thinking params, capped max_tokens to {}",
                            input_tokens_count,
                            current_max_tokens,
                            context_len,
                            retry_body["max_tokens"],
                        )
                        return retry_body
                    retry_body = clone_body_without_reasoning_budget(body)
                    if retry_body is not None:
                        retry_body["max_tokens"] = max(
                            context_len - input_tokens_count - 2048, 1
                        )
                        logger.warning(
                            "NIM_NONSTREAM: context overflow (input={} + max_tokens={} > context={}), "
                            "stripped reasoning_budget, capped max_tokens to {}",
                            input_tokens_count,
                            current_max_tokens,
                            context_len,
                            retry_body["max_tokens"],
                        )
                        return retry_body
                    if current_max_tokens > excess:
                        retry_body = {**body}
                        retry_body["max_tokens"] = max(
                            context_len - input_tokens_count - 2048, 1
                        )
                        logger.warning(
                            "NIM_NONSTREAM: context overflow (input={} + max_tokens={} > context={}), "
                            "thinking params already stripped, reducing max_tokens to {}",
                            input_tokens_count,
                            current_max_tokens,
                            context_len,
                            retry_body["max_tokens"],
                        )
                        return retry_body

        if "reasoning_budget" in error_text_lower:
            retry_body = clone_body_without_reasoning_budget(body)
            if retry_body is not None:
                logger.warning(
                    "NIM_STREAM: retrying without reasoning_budget after 400 error"
                )
                return retry_body

        if "chat_template" in error_text_lower:
            retry_body = clone_body_without_chat_template(body)
            if retry_body is not None:
                logger.warning(
                    "NIM_STREAM: retrying without chat_template after 400 error"
                )
                return retry_body

        if "reasoning_content" in error_text_lower:
            retry_body = clone_body_without_reasoning_content(body)
            if retry_body is not None:
                logger.warning(
                    "NIM_STREAM: retrying without reasoning_content after 400 error"
                )
                return retry_body

        thinking_hints = (
            "thinking",
            "enable_thinking",
            "chat_template_kwargs",
        )
        if any(hint in error_text_lower for hint in thinking_hints):
            retry_body = clone_body_without_all_thinking(body)
            if retry_body is not None:
                logger.warning(
                    "NIM_STREAM: 400 error mentions thinking params, retrying without all thinking extra_body fields"
                )
                return retry_body

        return None
