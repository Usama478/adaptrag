"""The real LLM: OpenAI's Responses API through the official `openai` client.

This is the only module that imports openai. The API key arrives as a plain string from
`provider.py` (taken out of SecretStr there) and is never logged or printed.
"""

import logging
from functools import lru_cache

from openai import APIError, OpenAI

from adaptrag.llm.base import LLM, LLMError

logger = logging.getLogger(__name__)

# The client retries connection errors, 408, 409, 429 and 5xx with a short backoff.
MAX_RETRIES = 2


class OpenAILLM:
    """Implements the `LLM` protocol with `client.responses.create`."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        timeout_seconds: float,
        max_output_tokens: int,
        temperature: float | None,
    ) -> None:
        self.model_name = model_name
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=MAX_RETRIES)

    def complete(self, system: str, user: str) -> str:
        try:
            # Two calls instead of one with **kwargs: temperature is sent only when it is
            # set, and mypy can still check every argument. No `reasoning` parameter:
            # gpt-4o-mini is not a reasoning model.
            if self._temperature is None:
                response = self._client.responses.create(
                    model=self.model_name,
                    instructions=system,
                    input=user,
                    max_output_tokens=self._max_output_tokens,
                )
            else:
                response = self._client.responses.create(
                    model=self.model_name,
                    instructions=system,
                    input=user,
                    max_output_tokens=self._max_output_tokens,
                    temperature=self._temperature,
                )
        except APIError as exc:
            # Every SDK error (timeout, connection, 4xx, 5xx) inherits from APIError.
            # The message names only the error type; the cause stays attached for logs.
            raise LLMError(f"OpenAI request failed ({type(exc).__name__})") from exc

        usage = response.usage
        if usage is None:
            logger.warning("LLM %s: token usage missing from the response", self.model_name)
        else:
            logger.info(
                "LLM %s: %d input tokens, %d output tokens",
                self.model_name,
                usage.input_tokens,
                usage.output_tokens,
            )

        text = response.output_text
        if not text.strip():
            raise LLMError(f"OpenAI returned no text (status: {response.status})")
        return text


@lru_cache
def get_openai_llm(
    api_key: str,
    model_name: str,
    timeout_seconds: float,
    max_output_tokens: int,
    temperature: float | None,
) -> LLM:
    """Return the client for these settings, built once per process (it keeps a connection
    pool, so building one per request would waste connections)."""
    return OpenAILLM(api_key, model_name, timeout_seconds, max_output_tokens, temperature)
