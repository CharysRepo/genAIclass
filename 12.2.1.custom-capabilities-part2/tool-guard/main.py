from dotenv import load_dotenv
import logfire
from pydantic_ai import Agent
from dataclasses import dataclass
from tool_guard import ToolGuard

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

@dataclass
class Deps:
    role: str

agent = Agent(
    'ollama:qwen3.5:9b',
    capabilities=[ToolGuard()],
    instructions="You are a helpful assistant."
)

async def main():
    deps = Deps(role="user")
    result = await agent.run("What's the temperature in london?", deps=deps)
    print(f"Output: {result.output}")
    result = await agent.run("What's the wind speed in london?", deps=deps)
    print(f"Output: {result.output}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())