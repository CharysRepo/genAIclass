import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
load_dotenv(override=True)
import logfire
from sliding_window import SlidingWindow
from compaction import Compaction
from session_persistence.persistence import SessionPersistence, RedisStorageService
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'ollama:qwen3.5:9b',
    capabilities=[
        SessionPersistence(RedisStorageService(host="localhost", port=6379)),
        #SlidingWindow(max_messages=4, keep_messages=2)
        Compaction(max_messages=6, keep_messages=2)
    ],
    system_prompt="You are a helpful assistant."
)

async def main():    
    print("Agent is ready! (Type 'exit' to quit)")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break
        result = await agent.run(user_input)
        print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
