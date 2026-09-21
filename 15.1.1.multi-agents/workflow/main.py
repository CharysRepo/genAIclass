import asyncio
from typing import Union
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire
from faq_agent import faq_agent, AutoResolution
from engg_agent import engineering_agent, EscalationTicket
from orchestrator_agent import orchestrator_agent

logfire.configure()
logfire.instrument_pydantic_ai()

async def dispatch_ticket(ticket_text: str) -> Union[AutoResolution, EscalationTicket]:
    print(f"\nUser Ticket: '{ticket_text}'")
    
    router_result = await orchestrator_agent.run(ticket_text)
    decision = router_result.output
    print(f"[Orchestrator Decision]: Handing off entirely to '{decision.target_agent.upper()}' agent.")
    
    shared_history = router_result.all_messages()
    
    if decision.target_agent == "faq":
        specialist_result = await faq_agent.run(
            "Address the ticket request present in the chat log history above.",
            message_history=shared_history
        )
        return specialist_result.output
                
    elif decision.target_agent == "engineering":
        specialist_result = await engineering_agent.run(
            "Diagnose the system error present in the chat log history above.",
            message_history=shared_history
        )
        return specialist_result.output
    
async def main():
    print("Support Agent is ready! (Type 'exit' to quit)")  
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ("exit", "quit"):
                break
            result = await dispatch_ticket(user_input)
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())

# I bought a pair of shoes yesterday but they are too small. How do I get my money back?
# Whenever I click checkout, the page freezes and throws a 503 Service Unavailable error on the console.