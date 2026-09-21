from typing import Any
from pydantic_ai import ModelMessage, RunContext, ModelRequestContext
from pydantic_ai.capabilities import AbstractCapability
from dataclasses import dataclass
from utils import _find_safe_cutoff

@dataclass
class SlidingWindow(AbstractCapability[Any]):
    max_messages: int = 100
    keep_messages: int = 40

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Trim the message list if it exceeds the configured threshold."""

        messages: list[ModelMessage] = list(request_context.messages)
        #print(len(messages))
        if len(messages) <= self.max_messages:
            return request_context

        cutoff = _find_safe_cutoff(messages, self.keep_messages)

        if cutoff > 0:
            request_context.messages = messages[cutoff:]

        return request_context