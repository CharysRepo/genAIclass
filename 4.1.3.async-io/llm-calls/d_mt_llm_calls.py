import concurrent
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
def run_agent(i):
    return agent.run_sync("what is the capital of india. Just answer in one word")

def main() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(run_agent, i) for i in range(100)]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    print(results)


start_time = time.perf_counter()
main()
end_time = time.perf_counter()
print(f"Time taken(Multi threading): {end_time - start_time:.2f} seconds")
