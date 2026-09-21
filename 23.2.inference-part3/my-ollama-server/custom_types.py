from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the author of this message (system, user, assistant).")
    content: str = Field(..., description="The contents of the message.")

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    do_sample: Optional[bool] = True
    temperature: Optional[float] = 0.7
    repetition_penalty:Optional[float] = 1.1
    max_tokens: Optional[int] = Field(default=100, ge=1, le=2048)