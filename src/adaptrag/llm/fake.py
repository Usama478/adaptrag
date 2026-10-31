"""A fake LLM for tests: a fixed reply, no network, no key, no cost.

It records what it was asked, so tests can check the prompt and whether it was called.
"""

from adaptrag.llm.base import LLMError


class FakeLLM:
    """Implements the `LLM` protocol with a canned reply."""

    def __init__(self, reply: str = "Fake answer [1].", raise_error: bool = False) -> None:
        self.model_name = "fake-llm"
        self.reply = reply
        self.raise_error = raise_error
        self.calls = 0
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        self.last_system = system
        self.last_user = user
        if self.raise_error:
            raise LLMError("FakeLLM was told to fail")
        return self.reply
