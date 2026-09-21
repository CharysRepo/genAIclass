from dotenv import load_dotenv
from pydantic_ai import Agent
import logfire
import time
import asyncio

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'ollama:llama3.1:8b',
    instructions="You are a helpful AI assistant",
)


async def main() -> None:
    for _ in range(10):
        res = await agent.run("what is the capital of india. Just answer in one word")
        print(res.output)


start_time = time.perf_counter()
asyncio.run(main())
end_time = time.perf_counter()
print(f"Time taken(Async-no batch): {end_time - start_time:.2f} seconds")
