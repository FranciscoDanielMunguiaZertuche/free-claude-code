import json
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from httpx import Request, Response

from config.nim import NimSettings
from core.anthropic.sse import SSEBuilder
from core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
)
from providers.defaults import NVIDIA_NIM_DEFAULT_BASE
from providers.nvidia_nim import NvidiaNimProvider
from providers.nvidia_nim.client import _convert_nonstream_to_sse
from providers.nvidia_nim.request import (
    NIM_TOOL_ARGUMENT_ALIASES_KEY,
    clone_body_without_all_thinking,
)


# Mock data classes
class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class MockBlock:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "test-model"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = ["STOP"]
        self.tools = []
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


def _input_json_deltas(events):
    deltas = []
    for event in events:
        if "event: content_block_delta" not in event:
            continue
        for line in event.splitlines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            delta = payload.get("delta", {})
            if delta.get("type") == "input_json_delta":
                deltas.append(delta.get("partial_json", ""))
    return deltas


def _tool_call_chunk(
    *,
    name,
    arguments,
    tool_id="call_1",
    index=0,
    finish_reason=None,
):
    mock_tc = MagicMock()
    mock_tc.index = index
    mock_tc.id = tool_id
    mock_tc.function.name = name
    mock_tc.function.arguments = arguments

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="", tool_calls=[mock_tc]),
            finish_reason=finish_reason,
        )
    ]
    mock_chunk.usage = None
    return mock_chunk


def _make_bad_request_error(message: str) -> openai.BadRequestError:
    response = Response(
        status_code=400,
        request=Request("POST", f"{NVIDIA_NIM_DEFAULT_BASE}/chat/completions"),
    )
    body = {"error": {"message": message, "type": "BadRequestError", "code": 400}}
    return openai.BadRequestError(message, response=response, body=body)


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        # execute_with_retry should call through to the actual function
        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.mark.asyncio
async def test_init(provider_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = NvidiaNimProvider(provider_config, nim_settings=NimSettings())
        assert provider._api_key == "test_key"
        assert provider._base_url == "https://test.api.nvidia.com/v1"
        mock_openai.assert_called_once()


@pytest.mark.asyncio
async def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""
    from providers.base import ProviderConfig

    config = ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        NvidiaNimProvider(config, nim_settings=NimSettings())
        call_kwargs = mock_openai.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


@pytest.mark.asyncio
async def test_build_request_body(provider_config):
    """Test request body construction."""
    provider = NvidiaNimProvider(provider_config, nim_settings=NimSettings())
    req = MockRequest()
    body = provider._build_request_body(req)

    assert body["model"] == "test-model"
    assert body["temperature"] == 0.5
    assert len(body["messages"]) == 2  # System + User
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "System prompt"

    assert "extra_body" in body
    ctk = body["extra_body"]["chat_template_kwargs"]
    assert ctk["thinking"] is True
    assert "reasoning_budget" not in body["extra_body"]


@pytest.mark.asyncio
async def test_build_request_body_omits_reasoning_when_globally_disabled(
    provider_config,
):
    provider = NvidiaNimProvider(
        provider_config.model_copy(update={"enable_thinking": False}),
        nim_settings=NimSettings(),
    )
    req = MockRequest()
    body = provider._build_request_body(req)

    extra = body.get("extra_body", {})
    assert "chat_template_kwargs" not in extra
    assert "reasoning_budget" not in extra


@pytest.mark.asyncio
async def test_build_request_body_omits_reasoning_when_request_disables_thinking(
    provider_config,
):
    provider = NvidiaNimProvider(provider_config, nim_settings=NimSettings())
    req = MockRequest()
    req.thinking.enabled = False
    body = provider._build_request_body(req)

    extra = body.get("extra_body", {})
    assert "chat_template_kwargs" not in extra
    assert "reasoning_budget" not in extra


def test_preflight_and_build_request_issue_206_post_tool_text(nim_provider):
    """Regression: assistant message with tool_use then text plus tool results (GitHub #206)."""
    tool_id = "toolu_issue_206"
    req = MockRequest(
        messages=[
            MockMessage("user", "Use echo once."),
            MockMessage(
                "assistant",
                [
                    MockBlock(
                        type="tool_use",
                        id=tool_id,
                        name="echo_smoke",
                        input={"value": "FCC_206"},
                    ),
                    MockBlock(
                        type="text",
                        text="Commentary after the tool row.",
                    ),
                ],
            ),
            MockMessage(
                "user",
                [
                    MockBlock(
                        type="tool_result", tool_use_id=tool_id, content="FCC_206"
                    ),
                    MockBlock(type="text", text="What was echoed?"),
                ],
            ),
        ],
    )
    nim_provider.preflight_stream(req, thinking_enabled=False)
    body = nim_provider._build_request_body(req, thinking_enabled=False)
    assert "messages" in body
    assert any(m.get("role") == "tool" for m in body["messages"])


@pytest.mark.asyncio
async def test_stream_response_text(nim_provider):
    """Test streaming text response."""
    req = MockRequest()

    # Create mock chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [
        MagicMock(
            delta=MagicMock(content="Hello", reasoning_content=""), finish_reason=None
        )
    ]
    mock_chunk1.usage = None

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [
        MagicMock(
            delta=MagicMock(content=" World", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk2.usage = MagicMock(completion_tokens=10)

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

        assert len(events) > 0
        assert "event: message_start" in events[0]

        text_content = ""
        for e in events:
            if "event: content_block_delta" in e and '"text_delta"' in e:
                for line in e.splitlines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "delta" in data and "text" in data["delta"]:
                            text_content += data["delta"]["text"]

        assert "Hello World" in text_content


@pytest.mark.asyncio
async def test_stream_response_thinking_reasoning_content(nim_provider):
    """Test streaming with native reasoning_content — NIM emits as text.

    NIM models with ``thinking=True`` place the entire response in
    ``reasoning_content`` (the ``content`` field is empty). The proxy
    therefore treats ``reasoning_content`` as regular text output.
    """
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="Thinking..."),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

        found_text = False
        for e in events:
            if "event: content_block_start" in e and '"text"' in e:
                found_text = True
        assert found_text
        event_text = "".join(events)
        assert "Thinking..." in event_text


@pytest.mark.asyncio
async def test_stream_response_suppresses_thinking_when_disabled(provider_config):
    provider = NvidiaNimProvider(
        provider_config.model_copy(update={"enable_thinking": False}),
        nim_settings=NimSettings(),
    )
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Answer", reasoning_content="Thinking..."),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in provider.stream_response(req)]

        event_text = "".join(events)
        assert "thinking_delta" not in event_text
        assert "Thinking..." in event_text
        assert "Answer" in event_text


def _make_bad_request_error(message: str) -> openai.BadRequestError:
    response = Response(status_code=400, request=Request("POST", "http://test"))
    body = {"error": {"message": message}}
    return openai.BadRequestError(message, response=response, body=body)


@pytest.mark.asyncio
async def test_stream_response_retries_without_chat_template(provider_config):
    provider = NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(chat_template="custom_template"),
    )
    req = MockRequest(model="mistralai/mixtral-8x7b-instruct-v0.1")

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="OK", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2)

    async def mock_stream():
        yield mock_chunk

    first_error = _make_bad_request_error(
        "chat_template is not supported for Mistral tokenizers."
    )

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [first_error, mock_stream()]

        events = [e async for e in provider.stream_response(req)]

    assert mock_create.await_count == 2

    first_extra = mock_create.call_args_list[0].kwargs["extra_body"]
    second_extra = mock_create.call_args_list[1].kwargs["extra_body"]

    assert first_extra["chat_template"] == "custom_template"
    assert first_extra["chat_template_kwargs"] == {
        "thinking": True,
    }
    assert "reasoning_budget" not in first_extra

    assert "chat_template" not in second_extra
    assert second_extra["chat_template_kwargs"] == {
        "thinking": True,
    }
    assert "reasoning_budget" not in second_extra

    event_text = "".join(events)
    assert "event: error" not in event_text
    assert "OK" in event_text


@pytest.mark.asyncio
async def test_stream_response_does_not_retry_unrelated_bad_request(provider_config):
    provider = NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(chat_template="custom_template"),
    )
    req = MockRequest(model="mistralai/mixtral-8x7b-instruct-v0.1")

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = _make_bad_request_error("unrelated bad request")

        events = [e async for e in provider.stream_response(req)]

    assert mock_create.await_count == 1
    event_text = "".join(events)
    assert "Invalid request sent to provider" in event_text
    assert "event: message_stop" in event_text


@pytest.mark.asyncio
async def test_tool_call_stream(nim_provider):
    """Test streaming tool calls."""
    req = MockRequest()

    # Mock tool call delta
    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "call_1"
    mock_tc.function.name = "search"
    mock_tc.function.arguments = '{"q": "test"}'

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="", tool_calls=[mock_tc]),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

        starts = [
            e for e in events if "event: content_block_start" in e and '"tool_use"' in e
        ]
        assert len(starts) == 1
        assert "search" in starts[0]


@pytest.mark.asyncio
async def test_stream_response_restores_aliased_tool_arguments(nim_provider):
    """NIM-safe argument aliases are restored before Anthropic SSE emission."""
    req = MockRequest(
        tools=[
            MockTool(
                "Grep",
                "Search file contents",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "-A": {"type": "number"},
                        "type": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            )
        ]
    )
    mock_chunk = _tool_call_chunk(
        name="Grep",
        arguments=json.dumps({"pattern": "needle", "-A": 2, "_fcc_arg_type": "py"}),
    )

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

    await_args = mock_create.await_args
    assert await_args is not None
    create_kwargs = await_args.kwargs
    assert NIM_TOOL_ARGUMENT_ALIASES_KEY not in create_kwargs
    properties = create_kwargs["tools"][0]["function"]["parameters"]["properties"]
    assert "-A" in properties
    assert "type" not in properties
    assert "_fcc_arg_A" not in properties
    assert "_fcc_arg_type" in properties

    deltas = _input_json_deltas(events)
    assert len(deltas) == 1
    assert json.loads(deltas[0]) == {"pattern": "needle", "-A": 2, "type": "py"}
    assert "_fcc_arg_type" not in deltas[0]


@pytest.mark.asyncio
async def test_stream_response_buffers_chunked_aliased_tool_arguments(nim_provider):
    """Chunked aliased args are emitted once as restored Claude Code args."""
    req = MockRequest(
        tools=[
            MockTool(
                "Grep",
                "Search file contents",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "type": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            )
        ]
    )
    first_chunk = _tool_call_chunk(
        name="Grep",
        arguments='{"pattern": "needle", ',
        tool_id="call_chunked",
    )
    second_chunk = _tool_call_chunk(
        name=None,
        arguments='"_fcc_arg_type": "py"}',
        tool_id="call_chunked",
    )

    async def mock_stream():
        yield first_chunk
        yield second_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

    deltas = _input_json_deltas(events)
    assert len(deltas) == 1
    assert json.loads(deltas[0]) == {"pattern": "needle", "type": "py"}


@pytest.mark.asyncio
async def test_stream_response_restores_nested_aliased_tool_arguments(nim_provider):
    req = MockRequest(
        tools=[
            MockTool(
                "NotionLike",
                "Nested type schema",
                {
                    "type": "object",
                    "properties": {
                        "parent": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "id": {"type": "string"},
                            },
                            "required": ["type", "id"],
                        }
                    },
                    "required": ["parent"],
                },
            )
        ]
    )
    mock_chunk = _tool_call_chunk(
        name="NotionLike",
        arguments=json.dumps(
            {"parent": {"_fcc_arg_type": "page_id", "id": "page_123"}}
        ),
    )

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

    deltas = _input_json_deltas(events)
    assert len(deltas) == 1
    assert json.loads(deltas[0]) == {"parent": {"type": "page_id", "id": "page_123"}}


@pytest.mark.asyncio
async def test_stream_response_task_tool_still_forces_background_false(nim_provider):
    req = MockRequest(
        tools=[
            MockTool(
                "Task",
                "Run a subagent",
                {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "prompt": {"type": "string"},
                        "run_in_background": {"type": "boolean"},
                    },
                    "required": ["description", "prompt"],
                },
            )
        ]
    )
    mock_chunk = _tool_call_chunk(
        name="Task",
        arguments=json.dumps(
            {
                "description": "Inspect",
                "prompt": "Read the marker",
                "run_in_background": True,
            }
        ),
        tool_id="call_task",
    )

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

    deltas = _input_json_deltas(events)
    assert len(deltas) == 1
    assert json.loads(deltas[0])["run_in_background"] is False


@pytest.mark.asyncio
async def test_stream_response_retries_without_reasoning_budget(nim_provider):
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    error = _make_bad_request_error("Unsupported field: enable_thinking")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [e async for e in nim_provider.stream_response(req)]

    assert mock_create.await_count == 2
    first_call = mock_create.await_args_list[0].kwargs
    second_call = mock_create.await_args_list[1].kwargs
    assert first_call["extra_body"]["chat_template_kwargs"]["thinking"] is True
    # On retry, all thinking-related fields should be stripped
    assert "chat_template_kwargs" not in second_call["extra_body"]
    assert any("Recovered" in event for event in events)
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_stream_response_retries_without_reasoning_content(nim_provider):
    req = MockRequest(
        system=None,
        messages=[
            MockMessage(
                "assistant",
                [
                    MockBlock(type="thinking", thinking="Need the tool."),
                    MockBlock(
                        type="tool_use",
                        id="toolu_reasoning",
                        name="echo_smoke",
                        input={"value": "FCC_TOOL"},
                    ),
                ],
            )
        ],
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    error = _make_bad_request_error("Unsupported field: reasoning_content")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [e async for e in nim_provider.stream_response(req)]

    assert mock_create.await_count == 2
    first_call = mock_create.await_args_list[0].kwargs
    second_call = mock_create.await_args_list[1].kwargs
    assert first_call["messages"][0]["reasoning_content"] == "Need the tool."
    assert "reasoning_content" not in second_call["messages"][0]
    assert second_call["messages"][0]["tool_calls"][0]["id"] == "toolu_reasoning"
    assert any("Recovered" in event for event in events)
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_stream_response_bad_request_without_reasoning_budget_does_not_retry(
    nim_provider,
):
    req = MockRequest()
    error = _make_bad_request_error("Unsupported field: top_k")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = error

        events = [e async for e in nim_provider.stream_response(req)]

    assert mock_create.await_count == 1
    assert any("Invalid request sent to provider" in event for event in events)
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_stream_response_bad_request_with_thinking_hint_strips_all(
    nim_provider,
):
    req = MockRequest()
    error = _make_bad_request_error("Unknown field: chat_template_kwargs")

    nonstream_response = _MockResponse(
        choices=[_MockChoice(_MockMessage(content="ok"))],
        usage=_MockUsage(10),
    )

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, nonstream_response]

        events = [e async for e in nim_provider.stream_response(req)]

        assert mock_create.await_count == 2
        assert any("event: content_block_start" in e for e in events)


@pytest.mark.asyncio
async def test_stream_response_bad_request_with_enable_thinking_hint_strips_all(
    nim_provider,
):
    req = MockRequest()
    error = _make_bad_request_error("Field enable_thinking is not supported")

    nonstream_response = _MockResponse(
        choices=[_MockChoice(_MockMessage(content="ok"))],
        usage=_MockUsage(10),
    )

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, nonstream_response]

        async for _ in nim_provider.stream_response(req):
            pass

        assert mock_create.await_count == 2


def test_clone_body_without_all_thinking_strips_all_thinking_fields():
    body = {
        "model": "moonshotai/kimi-k2.6",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32000,
        "extra_body": {
            "chat_template_kwargs": {
                "thinking": True,
            },
            "ignore_eos": False,
            "top_k": 50,
        },
    }
    result = clone_body_without_all_thinking(body)
    assert result is not None
    assert "chat_template_kwargs" not in result["extra_body"]
    assert "ignore_eos" not in result["extra_body"]
    assert result["extra_body"]["top_k"] == 50
    assert result["model"] == "moonshotai/kimi-k2.6"


def test_clone_body_without_all_thinking_returns_none_when_no_thinking_params():
    body = {
        "model": "moonshotai/kimi-k2.6",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32000,
        "extra_body": {"top_k": 50},
    }
    result = clone_body_without_all_thinking(body)
    assert result is None


def test_clone_body_without_all_thinking_removes_empty_extra_body():
    body = {
        "model": "moonshotai/kimi-k2.6",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32000,
        "extra_body": {"chat_template_kwargs": {"thinking": True}},
    }
    result = clone_body_without_all_thinking(body)
    assert result is not None
    assert "extra_body" not in result


class _MockFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _MockToolCall:
    def __init__(self, id: str | None, function: _MockFunction):
        self.id = id
        self.function = function


class _MockMessage:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list | None = None,
        reasoning_content: str | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _MockChoice:
    def __init__(self, message: _MockMessage, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason


class _MockUsage:
    def __init__(self, completion_tokens: int = 10):
        self.completion_tokens = completion_tokens


class _MockResponse:
    def __init__(self, choices: list, usage: _MockUsage | None = None):
        self.choices = choices
        self.usage = usage


def _parse_events(
    raw_events: list[str], prefix_events: list[str] | None = None
) -> list:
    all_events = (prefix_events or []) + raw_events
    raw_text = "".join(all_events)
    return parse_sse_text(raw_text)


def test_convert_nonstream_to_sse_text_only():
    message = _MockMessage(content="Hello world")
    response = _MockResponse([_MockChoice(message, "stop")], _MockUsage(5))
    sse = SSEBuilder("msg_test", "test-model", 10)
    prefix = [sse.message_start()]
    events = _convert_nonstream_to_sse(response, sse, thinking_enabled=False)
    parsed = _parse_events(events, prefix)
    assert_anthropic_stream_contract(parsed)
    assert any(
        e.event == "content_block_start"
        and e.data.get("content_block", {}).get("type") == "text"
        for e in parsed
    )
    text_parts = [
        e.data["delta"]["text"]
        for e in parsed
        if e.event == "content_block_delta"
        and e.data.get("delta", {}).get("type") == "text_delta"
    ]
    assert "Hello world" in "".join(text_parts)


def test_convert_nonstream_to_sse_tool_calls():
    tc = _MockToolCall("call_abc", _MockFunction("read_file", '{"path": "/tmp/x"}'))
    message = _MockMessage(content=None, tool_calls=[tc])
    response = _MockResponse([_MockChoice(message, "tool_calls")], _MockUsage(20))
    sse = SSEBuilder("msg_test", "test-model", 10)
    prefix = [sse.message_start()]
    events = _convert_nonstream_to_sse(response, sse, thinking_enabled=False)
    parsed = _parse_events(events, prefix)
    assert_anthropic_stream_contract(parsed)
    tool_starts = [
        e
        for e in parsed
        if e.event == "content_block_start"
        and e.data.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 1
    assert tool_starts[0].data["content_block"]["name"] == "read_file"
    assert tool_starts[0].data["content_block"]["id"] == "call_abc"
    json_deltas = [
        e
        for e in parsed
        if e.event == "content_block_delta"
        and e.data.get("delta", {}).get("type") == "input_json_delta"
    ]
    args_text = "".join(e.data["delta"]["partial_json"] for e in json_deltas)
    assert json.loads(args_text) == {"path": "/tmp/x"}


def test_convert_nonstream_to_sse_multiple_tool_calls():
    tc1 = _MockToolCall("call_1", _MockFunction("bash", '{"command": "ls"}'))
    tc2 = _MockToolCall("call_2", _MockFunction("read_file", '{"path": "/a"}'))
    message = _MockMessage(content=None, tool_calls=[tc1, tc2])
    response = _MockResponse([_MockChoice(message, "tool_calls")], _MockUsage(30))
    sse = SSEBuilder("msg_test", "test-model", 10)
    prefix = [sse.message_start()]
    events = _convert_nonstream_to_sse(response, sse, thinking_enabled=False)
    parsed = _parse_events(events, prefix)
    assert_anthropic_stream_contract(parsed)
    tool_starts = [
        e
        for e in parsed
        if e.event == "content_block_start"
        and e.data.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 2
    assert tool_starts[0].data["content_block"]["name"] == "bash"
    assert tool_starts[1].data["content_block"]["name"] == "read_file"


def test_convert_nonstream_to_sse_text_then_tool_calls():
    tc = _MockToolCall("call_1", _MockFunction("search", '{"q": "test"}'))
    message = _MockMessage(content="Let me search for that.", tool_calls=[tc])
    response = _MockResponse([_MockChoice(message, "tool_calls")], _MockUsage(25))
    sse = SSEBuilder("msg_test", "test-model", 10)
    prefix = [sse.message_start()]
    events = _convert_nonstream_to_sse(response, sse, thinking_enabled=False)
    parsed = _parse_events(events, prefix)
    assert_anthropic_stream_contract(parsed)
    block_starts = [e for e in parsed if e.event == "content_block_start"]
    types = [e.data["content_block"]["type"] for e in block_starts]
    assert "text" in types
    assert "tool_use" in types


def test_convert_nonstream_to_sse_reasoning_content():
    message = _MockMessage(content="Answer", reasoning_content="Deep thoughts")
    response = _MockResponse([_MockChoice(message, "stop")], _MockUsage(15))
    sse = SSEBuilder("msg_test", "test-model", 10)
    prefix = [sse.message_start()]
    events = _convert_nonstream_to_sse(response, sse, thinking_enabled=True)
    parsed = _parse_events(events, prefix)
    assert_anthropic_stream_contract(parsed)
    text_starts = [
        e
        for e in parsed
        if e.event == "content_block_start"
        and e.data.get("content_block", {}).get("type") == "text"
    ]
    assert len(text_starts) >= 1
    event_text = "".join(e.raw for e in parsed)
    assert "Deep thoughts" in event_text
    assert "Answer" in event_text


def test_convert_nonstream_to_sse_reasoning_suppressed_when_disabled():
    message = _MockMessage(content="Answer", reasoning_content="secret reasoning")
    response = _MockResponse([_MockChoice(message, "stop")], _MockUsage(15))
    sse = SSEBuilder("msg_test", "test-model", 10)
    prefix = [sse.message_start()]
    events = _convert_nonstream_to_sse(response, sse, thinking_enabled=False)
    parsed = _parse_events(events, prefix)
    assert_anthropic_stream_contract(parsed)
    event_text = "".join(e.raw for e in parsed)
    assert "thinking_delta" not in event_text
    assert "secret reasoning" in event_text
    assert "Answer" in event_text


def test_convert_nonstream_to_sse_empty_content_with_tool():
    tc = _MockToolCall("call_1", _MockFunction("run", '{"cmd": "true"}'))
    message = _MockMessage(content=None, tool_calls=[tc])
    response = _MockResponse([_MockChoice(message, "tool_calls")], _MockUsage(8))
    sse = SSEBuilder("msg_test", "test-model", 10)
    prefix = [sse.message_start()]
    events = _convert_nonstream_to_sse(response, sse, thinking_enabled=False)
    parsed = _parse_events(events, prefix)
    assert_anthropic_stream_contract(parsed)


def test_convert_nonstream_to_sse_tool_call_without_id_gets_uuid():
    tc = _MockToolCall(id=None, function=_MockFunction("my_tool", '{"x": 1}'))
    message = _MockMessage(content=None, tool_calls=[tc])
    response = _MockResponse([_MockChoice(message, "tool_calls")], _MockUsage(5))
    sse = SSEBuilder("msg_test", "test-model", 10)
    prefix = [sse.message_start()]
    events = _convert_nonstream_to_sse(response, sse, thinking_enabled=False)
    parsed = _parse_events(events, prefix)
    tool_starts = [
        e
        for e in parsed
        if e.event == "content_block_start"
        and e.data.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 1
    assert tool_starts[0].data["content_block"]["id"].startswith("tool_")


def test_convert_nonstream_to_sse_estimates_tokens_when_no_usage():
    message = _MockMessage(content="Short reply")
    response = _MockResponse([_MockChoice(message, "stop")], usage=None)
    sse = SSEBuilder("msg_test", "test-model", 10)
    prefix = [sse.message_start()]
    events = _convert_nonstream_to_sse(response, sse, thinking_enabled=False)
    parsed = _parse_events(events, prefix)
    assert_anthropic_stream_contract(parsed)
    deltas = [e for e in parsed if e.event == "message_delta"]
    assert len(deltas) >= 1
    assert deltas[-1].data["usage"]["output_tokens"] > 0


def test_convert_nonstream_to_sse_stop_reason_mapping():
    message = _MockMessage(content="Hi")
    response = _MockResponse([_MockChoice(message, "length")], _MockUsage(3))
    sse = SSEBuilder("msg_test", "test-model", 10)
    prefix = [sse.message_start()]
    events = _convert_nonstream_to_sse(response, sse, thinking_enabled=False)
    parsed = _parse_events(events, prefix)
    deltas = [e for e in parsed if e.event == "message_delta"]
    assert deltas[-1].data["delta"]["stop_reason"] == "max_tokens"


def test_convert_nonstream_to_sse_tool_calls_stop_reason():
    tc = _MockToolCall("call_1", _MockFunction("fn", "{}"))
    message = _MockMessage(content=None, tool_calls=[tc])
    response = _MockResponse([_MockChoice(message, "tool_calls")], _MockUsage(5))
    sse = SSEBuilder("msg_test", "test-model", 10)
    prefix = [sse.message_start()]
    events = _convert_nonstream_to_sse(response, sse, thinking_enabled=False)
    parsed = _parse_events(events, prefix)
    deltas = [e for e in parsed if e.event == "message_delta"]
    assert deltas[-1].data["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_stream_response_uses_streaming_when_tools_and_reasoning_content_is_text(
    nim_provider,
):
    req = MockRequest()
    req.tools = [MockTool("bash", "Run commands", {"type": "object"})]

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(
                        id="call_1",
                        index=0,
                        function=MagicMock(name="bash", arguments='{"command": "ls"}'),
                    )
                ],
            ),
            finish_reason="tool_calls",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    async def _passthrough(fn, *args, **kwargs):
        return await fn(*args, **kwargs)

    with (
        patch.object(
            nim_provider._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create,
        patch.object(
            nim_provider._global_rate_limiter,
            "execute_with_retry",
            new_callable=AsyncMock,
            side_effect=_passthrough,
        ),
    ):
        mock_create.return_value = mock_stream()
        [_ async for _ in nim_provider.stream_response(req)]
        mock_create.assert_awaited_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs.get("stream") is True


@pytest.mark.asyncio
async def test_stream_response_uses_nonstream_when_tools_and_reasoning_content_is_not_text(
    nim_provider,
):
    nim_provider._reasoning_content_is_text = False
    req = MockRequest()
    req.tools = [MockTool("bash", "Run commands", {"type": "object"})]

    tc = _MockToolCall("call_1", _MockFunction("bash", '{"command": "ls"}'))
    mock_message = _MockMessage(content=None, tool_calls=[tc])
    mock_response = _MockResponse(
        [_MockChoice(mock_message, "tool_calls")], _MockUsage(10)
    )

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response
        [_ async for _ in nim_provider.stream_response(req)]
        mock_create.assert_awaited_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs.get("stream") is False


@pytest.mark.asyncio
async def test_stream_response_uses_streaming_when_no_tools(nim_provider):
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Hello", reasoning_content=""), finish_reason="stop"
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    async def _passthrough(fn, *args, **kwargs):
        return await fn(*args, **kwargs)

    with (
        patch.object(
            nim_provider._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create,
        patch.object(
            nim_provider._global_rate_limiter,
            "execute_with_retry",
            new_callable=AsyncMock,
            side_effect=_passthrough,
        ),
    ):
        mock_create.return_value = mock_stream()
        [_ async for _ in nim_provider.stream_response(req)]
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs.get("stream") is True


@pytest.mark.asyncio
async def test_nonstream_tools_emits_error_on_rate_limit_exhausted(nim_provider):
    req = MockRequest()
    req.tools = [MockTool("bash", "Run commands", {"type": "object"})]

    rate_limit_error = openai.RateLimitError(
        "Too Many Requests",
        response=Response(
            status_code=429,
            request=Request("POST", f"{NVIDIA_NIM_DEFAULT_BASE}/chat/completions"),
        ),
        body={"error": {"message": "Too Many Requests", "type": "RateLimitError"}},
    )

    with (
        patch.object(
            nim_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ),
        patch.object(
            nim_provider._global_rate_limiter,
            "execute_with_retry",
            new_callable=AsyncMock,
            side_effect=rate_limit_error,
        ),
    ):
        events = [e async for e in nim_provider.stream_response(req)]

        event_text = "".join(events)
        assert "event: message_stop" in event_text
        assert "rate" in event_text.lower() or "too many" in event_text.lower()


@pytest.mark.asyncio
async def test_nonstream_tools_emits_error_when_both_paths_fail(nim_provider):
    req = MockRequest()
    req.tools = [MockTool("bash", "Run commands", {"type": "object"})]

    rate_limit_error = openai.RateLimitError(
        "Too Many Requests",
        response=Response(
            status_code=429,
            request=Request("POST", f"{NVIDIA_NIM_DEFAULT_BASE}/chat/completions"),
        ),
        body={"error": {"message": "Too Many Requests", "type": "RateLimitError"}},
    )

    with (
        patch.object(
            nim_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ),
        patch.object(
            nim_provider._global_rate_limiter,
            "execute_with_retry",
            new_callable=AsyncMock,
            side_effect=rate_limit_error,
        ),
    ):
        events = [e async for e in nim_provider.stream_response(req)]

        event_text = "".join(events)
        assert "event: message_stop" in event_text
        assert "rate" in event_text.lower() or "too many" in event_text.lower()


@pytest.mark.asyncio
async def test_nonstream_tools_emits_error_on_timeout_both_fail(nim_provider):
    req = MockRequest()
    req.tools = [MockTool("bash", "Run commands", {"type": "object"})]

    timeout_error = openai.APITimeoutError(request=Request("POST", "http://test"))

    with (
        patch.object(
            nim_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ),
        patch.object(
            nim_provider._global_rate_limiter,
            "execute_with_retry",
            new_callable=AsyncMock,
            side_effect=timeout_error,
        ),
    ):
        events = [e async for e in nim_provider.stream_response(req)]

        event_text = "".join(events)
        assert "event: message_stop" in event_text
        assert "provider api request failed" in event_text.lower()


@pytest.mark.asyncio
async def test_create_stream_timeout_retry_succeeds(nim_provider):
    """_create_stream retries once on timeout and returns the stream on retry."""
    req = MockRequest()
    timeout_error = openai.APITimeoutError(request=Request("POST", "http://test"))

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="hello", reasoning_content=None, tool_calls=None),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=1, prompt_tokens=5)
    mock_stream = MagicMock()
    mock_stream.__aiter__ = MagicMock(return_value=iter([mock_chunk]))

    call_count = 0

    async def _flip_side_effect(fn, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise timeout_error
        return mock_stream

    with (
        patch.object(
            nim_provider._global_rate_limiter,
            "execute_with_retry",
            new_callable=AsyncMock,
            side_effect=_flip_side_effect,
        ),
        patch("providers.nvidia_nim.client.asyncio.sleep", new_callable=AsyncMock),
    ):
        events = [e async for e in nim_provider.stream_response(req)]

    event_text = "".join(events)
    assert "event: message_stop" in event_text
    assert call_count == 2


@pytest.mark.asyncio
async def test_nonstream_tools_streaming_fallback_fixes_broken_tool_calls(
    nim_provider,
):
    nim_provider._reasoning_content_is_text = False
    req = MockRequest()
    req.tools = [MockTool("bash", "Run commands", {"type": "object"})]

    bad_request_error = openai.BadRequestError(
        "Bad Request",
        response=Response(
            status_code=400,
            request=Request("POST", f"{NVIDIA_NIM_DEFAULT_BASE}/chat/completions"),
        ),
        body={"error": {"message": "Unknown parameter: something"}},
    )

    mock_tc_first = MagicMock()
    mock_tc_first.index = 0
    mock_tc_first.id = "call_abc"
    mock_tc_first.type = "function"
    mock_tc_first.function = MagicMock()
    mock_tc_first.function.name = "bash"
    mock_tc_first.function.arguments = '{"comma'

    mock_tc_continuation = MagicMock()
    mock_tc_continuation.index = 0
    mock_tc_continuation.id = None
    mock_tc_continuation.type = None
    mock_tc_continuation.function = MagicMock()
    mock_tc_continuation.function.name = None
    mock_tc_continuation.function.arguments = 'nd": "ls"}'

    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [
        MagicMock(
            delta=MagicMock(
                content=None, reasoning_content="", tool_calls=[mock_tc_first]
            ),
            finish_reason=None,
        )
    ]
    mock_chunk1.usage = None

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [
        MagicMock(
            delta=MagicMock(
                content=None, reasoning_content="", tool_calls=[mock_tc_continuation]
            ),
            finish_reason="tool_calls",
        )
    ]
    mock_chunk2.usage = MagicMock(completion_tokens=10)

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    call_count = 0

    async def _execute_with_retry(fn, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise bad_request_error
        return await fn(*args, **kwargs)

    with (
        patch.object(
            nim_provider._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create,
        patch.object(
            nim_provider._global_rate_limiter,
            "execute_with_retry",
            new_callable=AsyncMock,
            side_effect=_execute_with_retry,
        ),
    ):
        mock_create.return_value = mock_stream()
        events = [e async for e in nim_provider.stream_response(req)]

        assert mock_create.await_count >= 1
        last_call = mock_create.call_args_list[-1]
        assert last_call.kwargs.get("stream") is True

    event_text = "".join(events)
    assert "event: message_stop" in event_text


@pytest.mark.asyncio
async def test_context_overflow_reduces_reasoning_budget(nim_provider):
    req = MockRequest()
    req.max_tokens = 32000
    req.tools = [MockTool("bash", "Run commands", {"type": "object"})]

    context_overflow_error = openai.BadRequestError(
        "You passed 170753 input tokens and requested 32000 output tokens. "
        "However, the model's context length is only 202752 tokens, "
        "resulting in a maximum input length of 170752 tokens. "
        "Please reduce the length of the input prompt. "
        "(parameter=input_tokens, value=170753)",
        response=Response(
            status_code=400,
            request=Request("POST", f"{NVIDIA_NIM_DEFAULT_BASE}/chat/completions"),
        ),
        body={
            "error": {
                "message": "You passed 170753 input tokens and requested 32000 output tokens. "
                "However, the model's context length is only 202752 tokens",
                "type": "BadRequestError",
                "param": "input_tokens",
                "code": 400,
            }
        },
    )

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="ok",
                tool_calls=[],
                reasoning_content=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(completion_tokens=5)

    async def _execute_with_retry(fn, *args, **kwargs):
        return mock_response

    with (
        patch.object(
            nim_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[context_overflow_error, mock_response],
        ),
        patch.object(
            nim_provider._global_rate_limiter,
            "execute_with_retry",
            new_callable=AsyncMock,
            side_effect=_execute_with_retry,
        ),
    ):
        events = [e async for e in nim_provider.stream_response(req)]

        event_text = "".join(events)
        assert "event: message_stop" in event_text


@pytest.mark.asyncio
async def test_context_overflow_second_retry_strips_all_thinking(nim_provider):
    nim_provider._reasoning_content_is_text = False
    req = MockRequest()
    req.max_tokens = 32000
    req.tools = [MockTool("bash", "Run commands", {"type": "object"})]

    context_overflow_error = openai.BadRequestError(
        "You passed 170753 input tokens and requested 32000 output tokens. "
        "However, the model's context length is only 202752 tokens, "
        "resulting in a maximum input length of 170752 tokens. "
        "Please reduce the length of the input prompt. "
        "(parameter=input_tokens, value=170753)",
        response=Response(
            status_code=400,
            request=Request("POST", f"{NVIDIA_NIM_DEFAULT_BASE}/chat/completions"),
        ),
        body={
            "error": {
                "message": "You passed 170753 input tokens and requested 32000 output tokens. "
                "However, the model's context length is only 202752 tokens",
                "type": "BadRequestError",
                "param": "input_tokens",
                "code": 400,
            }
        },
    )

    second_context_overflow = openai.BadRequestError(
        "You passed 170753 input tokens and requested 31935 output tokens. "
        "However, the model's context length is only 202752 tokens. "
        "(parameter=input_tokens, value=170753)",
        response=Response(
            status_code=400,
            request=Request("POST", f"{NVIDIA_NIM_DEFAULT_BASE}/chat/completions"),
        ),
        body={
            "error": {
                "message": "You passed 170753 input tokens and requested 31935 output tokens.",
                "type": "BadRequestError",
                "param": "input_tokens",
                "code": 400,
            }
        },
    )

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="ok after strip",
                tool_calls=[],
                reasoning_content=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(completion_tokens=5)

    call_count = 0

    async def _execute_with_retry(fn, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await fn(*args, **kwargs)

    with (
        patch.object(
            nim_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[
                context_overflow_error,
                second_context_overflow,
                mock_response,
            ],
        ),
        patch.object(
            nim_provider._global_rate_limiter,
            "execute_with_retry",
            new_callable=AsyncMock,
            side_effect=_execute_with_retry,
        ),
    ):
        events = [e async for e in nim_provider.stream_response(req)]

    event_text = "".join(events)
    assert "event: message_stop" in event_text
    assert call_count == 3


class TestRoundRobin:
    """Tests for round-robin load balancing across API keys."""

    def test_next_client_cycles_through_keys(self, provider_config):
        """_next_client() returns clients in round-robin order."""
        provider = NvidiaNimProvider(
            provider_config,
            nim_settings=NimSettings(),
            fallback_api_keys=["key_b", "key_c"],
        )
        all_clients = [provider._client, *provider._fallback_clients]
        assert len(all_clients) == 3

        results = [provider._next_client() for _ in range(6)]
        indices = [idx for _, idx in results]
        clients = [c for c, _ in results]

        assert indices == [0, 1, 2, 0, 1, 2]
        assert clients == [
            all_clients[0],
            all_clients[1],
            all_clients[2],
            all_clients[0],
            all_clients[1],
            all_clients[2],
        ]

    def test_next_client_single_key(self, provider_config):
        """_next_client() works with only the primary key (no fallbacks)."""
        provider = NvidiaNimProvider(provider_config, nim_settings=NimSettings())
        client, idx = provider._next_client()
        assert idx == 0
        assert client is provider._client

        client2, idx2 = provider._next_client()
        assert idx2 == 0
        assert client2 is provider._client

    @pytest.mark.asyncio
    async def test_streaming_round_robin_distributes_requests(self, provider_config):
        """Streaming path distributes requests across keys via round-robin."""
        provider = NvidiaNimProvider(
            provider_config,
            nim_settings=NimSettings(),
            fallback_api_keys=["key_b", "key_c"],
        )
        all_clients = [provider._client, *provider._fallback_clients]
        call_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

        async def make_streaming_request():
            req = MockRequest()
            mock_chunk = MagicMock()
            mock_chunk.choices = [
                MagicMock(
                    delta=MagicMock(content="ok", reasoning_content=""),
                    finish_reason="stop",
                )
            ]
            mock_chunk.usage = MagicMock(completion_tokens=2)

            async def mock_stream():
                yield mock_chunk

            async def _passthrough(fn, *args, **kwargs):
                return await fn(*args, **kwargs)

            with (
                patch.object(
                    provider._global_rate_limiter,
                    "execute_with_retry",
                    new_callable=AsyncMock,
                    side_effect=_passthrough,
                ),
            ):
                for c_idx, c in enumerate(all_clients):
                    if call_counts[c_idx] == 0:
                        with patch.object(
                            c.chat.completions, "create", new_callable=AsyncMock
                        ) as mock_create:
                            mock_create.return_value = mock_stream()
                            [_ async for _ in provider.stream_response(req)]
                            call_counts[c_idx] += 1
                        break

        for _ in range(3):
            await make_streaming_request()

        assert sum(call_counts.values()) == 3

    @pytest.mark.asyncio
    async def test_nonstream_round_robin_starts_with_rotated_key(self, provider_config):
        """Non-streaming (tools) path uses round-robin to pick starting key."""
        provider = NvidiaNimProvider(
            provider_config,
            nim_settings=NimSettings(),
            fallback_api_keys=["key_b", "key_c"],
        )

        _, first_idx = provider._next_client()
        assert first_idx == 0

        _, second_idx = provider._next_client()
        assert second_idx == 1

    @pytest.mark.asyncio
    async def test_round_robin_failover_still_works_streaming(self, provider_config):
        """Round-robin + failover: if the chosen key 429s, next key in rotation is tried."""
        provider = NvidiaNimProvider(
            provider_config,
            nim_settings=NimSettings(),
            fallback_api_keys=["key_b", "key_c"],
        )

        req = MockRequest()

        rate_limit_error = openai.RateLimitError(
            "Too Many Requests",
            response=Response(
                status_code=429,
                request=Request("POST", f"{NVIDIA_NIM_DEFAULT_BASE}/chat/completions"),
            ),
            body={"error": {"message": "Too Many Requests"}},
        )

        mock_chunk = MagicMock()
        mock_chunk.choices = [
            MagicMock(
                delta=MagicMock(content="recovered", reasoning_content=""),
                finish_reason="stop",
            )
        ]
        mock_chunk.usage = MagicMock(completion_tokens=2)

        async def mock_stream():
            yield mock_chunk

        call_count = 0

        async def _execute_with_retry(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise rate_limit_error
            return await fn(*args, **kwargs)

        with (
            patch.object(
                provider._global_rate_limiter,
                "execute_with_retry",
                new_callable=AsyncMock,
                side_effect=_execute_with_retry,
            ),
            patch.object(
                provider._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                side_effect=rate_limit_error,
            ),
            patch.object(
                provider._fallback_clients[0].chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=mock_stream(),
            ),
        ):
            events = [e async for e in provider.stream_response(req)]
            event_text = "".join(events)
            assert "recovered" in event_text or "event: message_stop" in event_text
