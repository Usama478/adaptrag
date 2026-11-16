"""A fake LLM for tests: canned replies, no network, no key, no cost.

It records what it was asked, so tests can check the prompt and whether it was called.
"""

from collections.abc import Sequence

from adaptrag.llm.base import LLMError, TokenUsage


class FakeLLM:
    """Implements the `LLM` protocol with canned replies.

    reply: returned by every call, unless `replies` still has entries.
    replies: returned one per call, in order; when they run out, `reply` is used again.
        This lets a test script a flow of several calls (generator, then judge).
    """

    def __init__(
        self,
        reply: str = "Fake answer [1].",
        raise_error: bool = False,
        replies: Sequence[str] | None = None,
    ) -> None:
        self.model_name = "fake-llm"
        self.reply = reply
        self.raise_error = raise_error
        self.calls = 0
        self.last_system: str | None = None
        self.last_user: str | None = None
        # Every user message so far, oldest first.
        self.users: list[str] = []
        self.last_usage: TokenUsage | None = None
        self._queue = list(replies) if replies is not None else []

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        self.last_system = system
        self.last_user = user
        self.users.append(user)
        if self.raise_error:
            self.last_usage = None
            raise LLMError("FakeLLM was told to fail")
        reply = self._queue.pop(0) if self._queue else self.reply
        # Word counts stand in for tokens, so usage tests have numbers they can predict.
        self.last_usage = TokenUsage(
            input_tokens=len(system.split()) + len(user.split()),
            output_tokens=len(reply.split()),
        )
        return reply
