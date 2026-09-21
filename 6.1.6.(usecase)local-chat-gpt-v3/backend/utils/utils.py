from typing import Any, Dict, List
from pydantic_ai import ModelMessage, ModelRequest, TextPart, UserPromptPart


def convert_pydanticai_to_openai(messages: List[ModelMessage]) -> List[Dict[str, Any]]:
    openai_messages = []
    for message in messages:
        if isinstance(message, ModelRequest):
            role = "user"
        else:
            role = "assistant" 
            
        content = ""
        for part in message.parts:
            if isinstance(part, (UserPromptPart, TextPart)):
                if part.content:
                    content += part.content

        if content.strip():
            openai_messages.append({"role": role, "content": content.strip()})

    return openai_messages