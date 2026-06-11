"""Logfare provider (OpenAI-compatible chat completions transport)."""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import LOGFARE_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body


class LogfareProvider(OpenAIChatTransport):
    """Provider for Logfare's OpenAI-compatible API (logfare.ai/v1)."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LOGFARE",
            base_url=config.base_url or LOGFARE_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
