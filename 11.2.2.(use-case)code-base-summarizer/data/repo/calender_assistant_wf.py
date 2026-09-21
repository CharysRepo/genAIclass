import asyncio
from typing import Union
from pydantic_ai import Agent
from models import DeleteInvite, MeetingInvite, UpdateInvite, RecurringMeetingInvite, RetrievalByDateInvite, RetrievalByTitleInvite, MissingInfo
from calender_service import CalenderService
from dotenv import load_dotenv
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    "ollama:glm-4.7-flash:q4_K_M",
    #'groq:llama-3.3-70b-versatile',
    output_type=Union[MeetingInvite, RecurringMeetingInvite,UpdateInvite, RetrievalByDateInvite, RetrievalByTitleInvite, DeleteInvite, MissingInfo],
    instructions="""
        You are a calendar assistant. Dont hallucinate and make up data. 
        If you dont have enough information to complete the task, ask for it clearly and concisely.   
    """
 )

async def main():
    calender_service = CalenderService()

    message_history = []
    print("Calendar Agent is ready! (Type 'exit' to quit)")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']: break 

        try:
            result = await agent.run(user_input, message_history=message_history)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            continue

        if isinstance(result.output, MissingInfo):
            print(f"AI: {result.output.question}")
            message_history = result.all_messages() 
        
        elif isinstance(result.output, MeetingInvite):
            result = calender_service.create_google_invite(result.output)
            print(result)
            message_history = []

        elif isinstance(result.output, RecurringMeetingInvite):
            result = calender_service.create_google_invite(result.output)
            print(result)
            message_history = []

        elif isinstance(result.output, UpdateInvite):
            result = calender_service.update_google_invite(result.output)
            print(result)
            message_history = []
        elif isinstance(result.output, RetrievalByDateInvite):
            result = calender_service.list_google_calender_events_by_date(result.output)
            print(result)
            message_history = []
        elif isinstance(result.output, RetrievalByTitleInvite):
            result = calender_service.list_google_calender_events_by_title(result.output)
            print(result)
            message_history = []
        elif isinstance(result.output, DeleteInvite):
            result = calender_service.delete_google_invite_silently(result.output)
            print(result)
            message_history = []

if __name__ == "__main__":
    asyncio.run(main())

# create a new meeting invite on 6th april 2026 at 10am IST with title "ProjectTest" and attendees algorithmica.desktop@gmail.com