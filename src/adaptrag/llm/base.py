"""The LLM seam: what the rest of the app may ask of a text-generation model.

Code outside `llm/` depends on this protocol, never on the openai package, so the
provider can change (or be faked in tests) without touching callers.
"""

from typing import Protocol


class LLMError(Exception):
    """The model could not produce an answer: timeout, API error, or empty output.

    The message is for logs; users see a short generic message (502).
    """


class LLM(Protocol):
    """Turns a system prompt and a user message into one text reply.

    model_name: identifies the model (shown in logs).
    """

    model_name: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's reply. Raises LLMError if there is no usable reply."""
        ...
