import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
load_dotenv(override=True)
import logfire
from ui_event_streaming import StreamTransformer

logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'ollama:qwen3.5:9b',
    capabilities=[StreamTransformer()],
    instructions="You are a helpful assistant."
)

@agent.tool
def get_weather(ctx:RunContext[None], location: str) -> str:
    return f"The weather in {location} is 22°C."


async def main():    
    print("Agent is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break

        event_stream = agent.run_stream_events(user_input)
        async for event in event_stream:
            print(f"Received event: {event}")    
if __name__ == "__main__":
    asyncio.run(main())