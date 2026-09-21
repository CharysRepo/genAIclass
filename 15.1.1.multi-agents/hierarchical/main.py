import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire
from orchestrator_agent import orchestrator_agent

logfire.configure()
logfire.instrument_pydantic_ai()


async def main():
    print("Support Agent is ready! (Type 'exit' to quit)")  
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ("exit", "quit"):
                break
            result = await orchestrator_agent.run(user_input)
            print(result.output.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())

# I bought a pair of shoes yesterday but they are too small. How do I get my money back?
# Whenever I click checkout, the page freezes and throws a 503 Service Unavailable error on the console.