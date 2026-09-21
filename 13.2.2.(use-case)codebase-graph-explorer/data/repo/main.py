import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire
from services.hybrid_search_service import HybridSearchService
from db.database_manager import DatabaseManager
from agents.rag_agent import rag_agent, KnowledgeDeps

logfire.configure()
logfire.instrument_pydantic_ai()

async def main():
    await DatabaseManager.initialize()
    hybrid_search_service = HybridSearchService()
    deps = KnowledgeDeps(hybrid_search_service)

    message_history = []
    
    print("RAG Agent is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            result = await rag_agent.run(
                user_input, 
                deps=deps,
                message_history=message_history
            )
        except Exception as e:
            print(f"Error: {e}")
            continue
        message_history = result.all_messages()   
        response = result.output     
        print(response)
    await DatabaseManager.disconnect()

if __name__ == "__main__":
    asyncio.run(main())

# Can I use ChatGPT for my assignment?
# What happens if I get sick during an exam?
# How is the AI & Machine Learning course graded?