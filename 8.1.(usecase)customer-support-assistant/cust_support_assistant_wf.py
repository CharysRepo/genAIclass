import asyncio
from pydantic_ai import Agent
from models import Ticket
from utils import get_email_content
from email_service import EmailService
from db_service import DBService
from dotenv import load_dotenv
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

support_agent = Agent(
    #"ollama:llama3-groq-tool-use:8b",
    #"groq:llama-3.3-70b-versatile",
    "ollama:glm-4.7-flash:q4_K_M",
    output_type= str | Ticket,
    instructions= """
    You are an ISP support agent. Extract details and return a structured ticket. 
    If required data is missing for any intent, do not guess and makeup data. If you dont have enough information to complete the task, ask for it clearly and concisely.
    """
)

async def main():
    message_history = []
    
    print("Customer Support Agent is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break

        try:
            result = await support_agent.run(
                user_input, 
                message_history=message_history
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        if isinstance(result.output, str):
            print(f"AI: {result.output}")
            message_history = result.all_messages() 
        
        elif isinstance(result.output, Ticket):
            print(result.output)
            email_service = EmailService()
            email_details = get_email_content(result.output)  
            email_service.send_email(email_details)

            db_service = DBService()
            db_service.save_ticket(result.output)
            message_history = []

if __name__ == "__main__":
    asyncio.run(main())

# i've been waiting 4 hours and no one helped. fix my internet now! id: 99, email: algorithmica.desktop@gmail.com
