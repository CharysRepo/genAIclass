import re
from typing import Any
from pydantic_ai import RunContext, ModelRequestContext, UserPromptPart
from pydantic_ai.capabilities import AbstractCapability
from dataclasses import dataclass

@dataclass
class PIIRedactionGuardrail(AbstractCapability[Any]):
    
    async def before_model_request(self, ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        
        for message in request_context.messages:
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    part.content = re.sub(email_pattern, "[REDACTED_EMAIL]", part.content)
                    part.content = re.sub(phone_pattern, "[REDACTED_PHONE]", part.content)
        
        return request_context


