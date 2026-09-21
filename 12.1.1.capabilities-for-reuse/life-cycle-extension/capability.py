from typing import Any
from dotenv import load_dotenv
import logfire
from pydantic_ai import Agent, AgentRunResult, ModelResponse, RunContext, ModelRequestContext
from pydantic_ai.capabilities import AbstractCapability
from dataclasses import dataclass

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

@dataclass
class AuditLogger(AbstractCapability[Any]):

    async def before_model_request(self, ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        print("--- [BEFORE REQUEST] ---")
        return request_context

    async def after_model_request(self, ctx: RunContext[None], response: ModelResponse, request_context: ModelRequestContext) -> ModelResponse:
        print("--- [AFTER RESPONSE] ---")
        return response
    
    async def before_tool_execute(self, ctx: RunContext[None], *, args: dict, call: object, tool_def: object) -> dict:
        print("--- [BEFORE TOOL EXECUTION] ---")
        return args
    
    async def after_tool_execute(self, ctx: RunContext[None], *, args: dict, call: object, tool_def: object, result: Any) -> Any:
        print("--- [AFTER TOOL EXECUTION] ---")
        return result

    async def before_tool_validate(self, ctx: RunContext[None], *, args: dict, call: object, tool_def: object) -> object:
        print("--- [BEFORE TOOL VALIDATION] ---")
        return args

    async def after_tool_validate(self, ctx: RunContext[None], *, args: dict, call: object, tool_def: object) -> object:
        print("--- [AFTER TOOL VALIDATION] ---")
        return args

    async def before_run(self, ctx: RunContext[None]) -> None:
        print("--- [BEFORE RUN] ---")

    async def after_run(self, ctx: RunContext[None], result: AgentRunResult) -> AgentRunResult:
        print("--- [AFTER RUN] ---")
        return result

agent = Agent(
    'ollama:qwen3.5:9b',
    capabilities=[AuditLogger()],
    instructions="Check weather using the tool get_weather."
)

@agent.tool
def get_weather(ctx:RunContext[None], location: str) -> str:
    return f"The weather in {location} is 22°C."


async def main():
    result = await agent.run("What's the weather in london?")
    print(f"Output: {result.output}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())