from dataclasses import dataclass
from typing import Any, AsyncIterable, Union
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai import RunContext, AgentStreamEvent, AgentRunResultEvent
from pydantic_ai.messages import (
    ThinkingPart, 
    ThinkingPartDelta,
    TextPart, 
    TextPartDelta,
    PartStartEvent, 
    PartDeltaEvent, 
    FunctionToolCallEvent, 
    FunctionToolResultEvent
)

@dataclass
class StreamTransformer(AbstractCapability[Any]):
    async def wrap_run_event_stream(
        self,
        ctx: RunContext[Any],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[Union[AgentStreamEvent, dict[str, Any]]]:
        async for event in stream:
            # 1. Handle Start Events (Initial content)
            if isinstance(event, PartStartEvent):
                if isinstance(event.part, TextPart) and event.part.content:
                    yield {"type": "text", "content": event.part.content}
                elif isinstance(event.part, ThinkingPart) and event.part.content:
                    yield {"type": "thought", "content": event.part.content}

            # 2. Handle Delta Events (Streaming chunks)
            elif isinstance(event, PartDeltaEvent):
                if isinstance(event.delta, TextPartDelta):
                    yield {"type": "text", "content": event.delta.content_delta}
                elif isinstance(event.delta, ThinkingPartDelta):
                    yield {"type": "thought", "content": event.delta.content_delta}

            # 3. Handle Tool Calls
            elif isinstance(event, FunctionToolCallEvent):
                yield {
                    "type": "tool_call",
                    "tool_name": event.part.tool_name,
                    "tool_call_id": event.part.tool_call_id,
                    "args": event.part.args
                }

            # 4. Handle Tool Results
            elif isinstance(event, FunctionToolResultEvent):
                yield {
                    "type": "tool_result",
                    "tool_name": event.result.tool_name,
                    "tool_call_id": event.result.tool_call_id,
                    "result": str(event.result.content)
                }
                
            # 5. Handle Final Run Result 
            elif isinstance(event, AgentRunResultEvent):
                continue
