import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire
from agents.code_agent import code_agent
from config.config_reader import settings

logfire.configure()
logfire.instrument_pydantic_ai()

async def main():

    print("Code Agent is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            result = await code_agent.run(
                user_input
            )
        except Exception as e:
            print(f"Error: {e}")
            continue  
        response = result.output     
        print(response)

if __name__ == "__main__":
    asyncio.run(main())
