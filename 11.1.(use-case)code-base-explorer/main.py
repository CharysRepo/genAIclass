import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire
from services.hybrid_search_service import HybridSearchService
from db.database_manager import DatabaseManager
from agents.code_explorer_agent import code_explorer_agent, KnowledgeDeps

logfire.configure()
logfire.instrument_pydantic_ai()

async def main():
    await DatabaseManager.initialize()
    hybrid_search_service = HybridSearchService()
    deps = KnowledgeDeps(hybrid_search_service)

    message_history = []
    
    print("Code Agent is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            result = await code_explorer_agent.run(
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

# what does hybrdi search service do?
# can you explain ingestion service?
# can you explain rrf?