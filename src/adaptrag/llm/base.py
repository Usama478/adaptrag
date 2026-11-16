"""The LLM seam: what the rest of the app may ask of a text-generation model.

Code outside `llm/` depends on this protocol, never on the openai package, so the
provider can change (or be faked in tests) without touching callers.
"""

from dataclasses import dataclass
from typing import Protocol


class LLMError(Exception):
    """The model could not produce an answer: timeout, API error, or empty output.

    The message is for logs; users see a short generic message (502).
    """


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """The tokens one call used, as reported by the provider."""

    input_tokens: int
    output_tokens: int


class LLM(Protocol):
    """Turns a system prompt and a user message into one text reply.

    model_name: identifies the model (shown in logs).
    last_usage: tokens of the most recent `complete` call, or None if the provider did
        not report them (or the call failed before a response arrived). Evaluation
        scripts read it to keep the usage log (evaluation/usage.py).
    """

    model_name: str
    last_usage: TokenUsage | None

    def complete(self, system: str, user: str) -> str:
        """Return the model's reply. Raises LLMError if there is no usable reply."""
        ...
