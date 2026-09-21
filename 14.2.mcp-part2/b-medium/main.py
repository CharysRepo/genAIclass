import asyncio
import logging
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire
from math_agent import math_agent

logfire.configure()
logfire.instrument_pydantic_ai()

mcp_logger = logging.getLogger("mcp")
mcp_logger.setLevel(logging.DEBUG)
mcp_logger.addHandler(logfire.LogfireLoggingHandler()) 

async def main():
    print("Test Agent is ready! (Type 'exit' to quit)")  
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ("exit", "quit"):
                break
            result = await math_agent.run(user_input)
            print(result.output)
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    asyncio.run(main())

# calculate the area of traingle whole base is 6 and height is 8
# calculate the length of hypotenuse of triangle whole base is 6 and height is 8
