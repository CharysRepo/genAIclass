from dotenv import load_dotenv
import logfire
from pydantic_ai import Agent, RunContext

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'ollama:qwen3.5:9b',
    instructions="You are a helpful assistant."
)

@agent.tool
def get_weather(ctx:RunContext[None], location: str) -> str:
    return f"The weather in {location} is 22°C."


async def main():
    user_input = "What's the weather in paris?"
    async with agent.iter(user_input) as agent_run:
        async for node in agent_run:
            print(node)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())