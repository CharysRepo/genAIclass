from dotenv import load_dotenv
import logfire
from pydantic_ai import Agent, RunContext, ToolDefinition
from dataclasses import dataclass

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

@dataclass
class Deps:
    role: str

async def role_based_select(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    if ctx.deps.role == "user":
        return [tool_def for tool_def in tool_defs if tool_def.name == "get_temperature"]
    elif ctx.deps.role == "admin":
        return tool_defs
    else:
        return None
        
agent = Agent(
    'ollama:qwen3.5:9b',
    prepare_tools=role_based_select,
    instructions="You are a helpful assistant."
)

@agent.tool
def get_temperature(ctx:RunContext[Deps],city: str) -> str:
    """Get the current temperature for a city."""
    return f"The temperature in {city} is 22°C."

@agent.tool
def get_wind_speed(ctx:RunContext[Deps], city: str) -> str:
    """Get the current wind speed for a city in km/h."""
    return f"The wind speed in {city} is 15 km/h."



async def main():
    deps = Deps(role="user")
    result = await agent.run("What's the temperature in london?", deps=deps)
    print(f"Output: {result.output}")
    result = await agent.run("What's the wind speed in london?", deps=deps)
    print(f"Output: {result.output}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())