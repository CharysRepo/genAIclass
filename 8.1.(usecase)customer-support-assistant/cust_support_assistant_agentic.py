import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from models import Ticket
from email_service import EmailService
from utils import get_email_content
from db_service import DBService
from dotenv import load_dotenv
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

@dataclass
class Deps:
    email_service: EmailService
    db_service: DBService
    ticket_state: Ticket
    
support_agent = Agent(
    #"ollama:llama3-groq-tool-use:8b",
    #"groq:llama-3.3-70b-versatile",
    "ollama:glm-4.7-flash:q4_K_M",
    instructions="""
    You are an ISP support agent. Follow these rules:
    1. Strictly use tools in sequential order: gather_ticket_details -> send_email -> store_in_db. Do not skip steps or change the order.   
    2. If required data is missing for any intent, do not guess and makeup data. If you dont have enough information to complete the task, ask for it clearly and concisely.
    """
)

@support_agent.tool
def gather_ticket_details(ctx: RunContext[Deps], ticket: Ticket) -> str:
    """Gathers details for a new support ticket.""" 
    ctx.deps.ticket_state = ticket   
    return "gathered ticket details"

@support_agent.tool
def send_email(ctx: RunContext[Deps]) -> str:
    """Sends an email to the user."""
    ctx.deps.email_service.send_email(get_email_content(ctx.deps.ticket_state))
    return "email sent"

@support_agent.tool
def store_in_db(ctx: RunContext[Deps]) -> str:
    """Stores the ticket in the database."""
    ctx.deps.db_service.save_ticket(ctx.deps.ticket_state)
    return "ticket stored"

async def main():
    deps = Deps(
        email_service=EmailService(),
        db_service=DBService(),
        ticket_state=None
    )
    message_history = []
    
    print("Customer Support Agent is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break

        try:
            result = await support_agent.run(
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

# i've been waiting 4 hours and no one helped. fix my internet now! id: 99, email: algorithmica.desktop@gmail.com
