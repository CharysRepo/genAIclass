from dotenv import load_dotenv
from pydantic_ai import Agent
import logfire
import time

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are a helpful AI assistant",
)


def main() -> None:
    for _ in range(100):
        res = agent.run_sync("what is the capital of india. Just answer in one word")
        print(res.output)


start_time = time.perf_counter()
main()
end_time = time.perf_counter()
print(f"Time taken(sync): {end_time - start_time:.2f} seconds")
