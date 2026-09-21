import asyncio
from pydantic_ai import Agent
from pydantic_ai import RunContext
from calender_service import CalenderService
from models import DeleteInvite, MeetingInvite, RecurringMeetingInvite, RetrievalByTitleInvite, UpdateInvite, RetrievalByDateInvite, Deps
from dotenv import load_dotenv
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()


calender_agent = Agent(
    #'groq:llama-3.3-70b-versatile',
    #'github:openai/gpt-4o',
    "ollama:glm-4.7-flash:q4_K_M",
    deps_type=Deps,
    instructions="""
    You are a calendar assistant. Follow these rules:
    1. When adding or removing users, use the update_invite tool.
    2. When retrieving events, use the list_events_by_date or list_events_by_title tools as appropriate.
    3. If required data is missing for any intent, do not guess and makeup data. If you dont have enough information to complete the task, ask for it clearly and concisely.
    """
)


    
@calender_agent.tool
def create_invite(ctx: RunContext[Deps], invite: MeetingInvite) -> str:
    """Creates a brand new event on the user's primary calendar."""   
    return ctx.deps.calender_service.create_google_invite(invite)


@calender_agent.tool
def create_recurring_invite(ctx: RunContext[Deps], invite: RecurringMeetingInvite) -> str:
    """Creates a recurring event on the user's primary calendar."""   
    return ctx.deps.calender_service.create_google_invite(invite)

@calender_agent.tool
def update_invite(ctx: RunContext[Deps], update: UpdateInvite) -> str:
    """Add or remove attendees to an existing meeting."""
    return ctx.deps.calender_service.update_google_invite(update)

@calender_agent.tool
def list_events_by_date(ctx: RunContext[Deps], request: RetrievalByDateInvite) -> str:
    """Retrieves all events for a specific year or month."""
    return ctx.deps.calender_service.list_google_calender_events_by_date(request)

@calender_agent.tool
def list_events_by_title(ctx: RunContext[Deps], request: RetrievalByTitleInvite) -> str:
    """Retrieves all events for a specific title."""
    return ctx.deps.calender_service.list_google_calender_events_by_title(request)

@calender_agent.tool
def delete_invite(ctx: RunContext[Deps], event_id: DeleteInvite) -> str:
    """Deletes an existing meeting from the calendar."""
    return ctx.deps.calender_service.delete_google_invite_silently(event_id)

async def main():
    deps = Deps(calender_service=CalenderService())
    message_history = []
    
    print("Calendar Agent is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break

        try:
            result = await calender_agent.run(
                user_input, 
                deps=deps,
                message_history=message_history
            )
        except Exception as e:
            print(f"Error: {e}")
            continue
        message_history = result.all_messages()        
        print(f"Agent: {result.output}")

if __name__ == "__main__":
    asyncio.run(main())

# create a new recurring meeting invite on every saturday and sunday from 11th april 2026 to 31st april 2026. Timing:7.30am to 10am. title:"GenerativeAI Sessions" timezone:Asia/Kolkata emails=algorithmica.desktop@gmail.com

# create a new meeting invite with title "Interview with Google" on 15th april 2026 from 3pm to 4pm IST and add attendees with emails: algorithmica.desktop@gmail.com