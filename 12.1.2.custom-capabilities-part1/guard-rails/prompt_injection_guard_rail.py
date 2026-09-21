import re
from typing import Any
from pydantic_ai import RunContext, ModelRequestContext
from pydantic_ai.capabilities import AbstractCapability
from dataclasses import dataclass
from pydantic_ai.messages import UserPromptPart

@dataclass
class PromptInjectionGuardrail(AbstractCapability[Any]):
    
    async def before_model_request(self, ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        blocked_phrases = ["ignore all previous instructions", "system prompt", "developer mode"]
        
        for message in request_context.messages:
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    content_lower = part.content.lower()
                    if any(phrase in content_lower for phrase in blocked_phrases):
                        raise ValueError(f"Prompt injection detected in input: '{part.content}'")
        
        return request_context


