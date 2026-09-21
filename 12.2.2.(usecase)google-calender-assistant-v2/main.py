from dotenv import load_dotenv
load_dotenv(override=True)
import logfire
import asyncio
from calender_service import CalenderService
from calender_agent import calender_agent, Deps

logfire.configure()
logfire.instrument_pydantic_ai()

async def main():
    deps = Deps(calender_service=CalenderService())

    print("Calendar Agent is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break
        result = await calender_agent.run(
            user_input, 
            deps=deps
        )        
        print(f"Agent: {result.output}")

if __name__ == "__main__":
    asyncio.run(main())

# create a new recurring meeting invite on every saturday and sunday from 11th april 2026 to 31st april 2026. Timing:7.30am to 10am. title:"GenerativeAI Sessions" timezone:Asia/Kolkata emails=algorithmica.desktop@gmail.com

# create a new meeting invite with title "Interview with Google" on 11th May 2026 from 3pm to 4pm and add attendees with emails: algorithmica.desktop@gmail.com