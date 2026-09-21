import asyncio

from dotenv import load_dotenv
from pydantic_ai import Agent
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are helpful assistant."
)

async def main(query):
    async with agent.run_stream(query) as result:
        async for text in result.stream_text(delta=True):
            print(text)


asyncio.run(main("What are your capabilities?"))