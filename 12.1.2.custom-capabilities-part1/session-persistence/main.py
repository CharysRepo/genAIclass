import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
load_dotenv(override=True)
import logfire
from session_persistence import RedisStorageService, SessionPersistence

logfire.configure()
logfire.instrument_pydantic_ai()

redis_storage_service = RedisStorageService(host="localhost", port=6379)
session_persistence = SessionPersistence(
    store=redis_storage_service
)

agent = Agent(
    'ollama:qwen3.5:9b',
    capabilities=[session_persistence],
    instructions="You are a helpful assistant."
)

async def main():    
    print("Agent is ready! (Type 'exit' to quit)")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            result = await agent.run(user_input)
        except Exception as e:
            print(f"Error: {e}")
            continue     
        print(result.output)

if __name__ == "__main__":
    asyncio.run(main())