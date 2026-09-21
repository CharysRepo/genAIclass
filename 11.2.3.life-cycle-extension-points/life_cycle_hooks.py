from typing import Any
from dotenv import load_dotenv
import logfire
from pydantic_ai import Agent, AgentRunResult, ModelResponse, RunContext, ModelRequestContext
from pydantic_ai.capabilities import Hooks

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

hooks = Hooks()

@hooks.on.before_model_request
async def log_request(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
    print("--- [BEFORE REQUEST] ---")
    return request_context

@hooks.on.after_model_request
async def log_response(ctx: RunContext[None], response: ModelResponse, request_context: ModelRequestContext) -> ModelResponse:
    print("--- [AFTER RESPONSE] ---")
    return response

@hooks.on.before_tool_execute
async def before_tool_exec(ctx: RunContext[None], *, args: dict, call: object, tool_def: object) -> dict:
    print("--- [BEFORE TOOL EXECUTION] ---")
    return args

@hooks.on.after_tool_execute
async def after_tool_exec(ctx: RunContext[None], *, args: dict, call: object, tool_def: object, result: Any) -> Any:
    print("--- [AFTER TOOL EXECUTION] ---")
    return result

@hooks.on.before_tool_validate
async def before_tool_validate(ctx: RunContext[None], *, args: dict, call: object, tool_def: object) -> object:
    print("--- [BEFORE TOOL VALIDATION] ---")
    return args

@hooks.on.after_tool_validate
async def after_tool_validate(ctx: RunContext[None], *, args: dict, call: object, tool_def: object) -> object:
    print("--- [AFTER TOOL VALIDATION] ---")
    return args

@hooks.on.before_run
async def before_run(ctx: RunContext[None]) -> None:
    print("--- [BEFORE RUN] ---")

@hooks.on.after_run
async def after_run(ctx: RunContext[None], result: AgentRunResult) -> AgentRunResult:
    print("--- [AFTER RUN] ---")
    return result

agent = Agent(
    'ollama:qwen3.5:9b',
    capabilities=[hooks],
    instructions="Check weather using the tool get_weather."
)

@agent.tool
def get_weather(ctx:RunContext[None], location: str) -> str:
    print("--- [TOOL] get_weather called ---")
    return f"The weather in {location} is 22°C."


async def main():
    user_input = "What's the weather in paris?"
    async with agent.iter(user_input) as agent_run:
        async for node in agent_run:
            print(node)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())