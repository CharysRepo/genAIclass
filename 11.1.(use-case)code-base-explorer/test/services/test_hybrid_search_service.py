from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire
import asyncio
from db.database_manager import DatabaseManager
from services.hybrid_search_service import HybridSearchService

logfire.configure()
logfire.instrument_pydantic_ai()

async def main():
    await DatabaseManager.initialize()
    hybrid_search_service = HybridSearchService()     
    while True:
        print("\nEnter a query (or 'exit' to quit):")
        user_input = input("> ")
        if user_input.lower() == "exit":
            break
        result = await hybrid_search_service.search(user_input, limit= 5)
        print(result)
    await DatabaseManager.disconnect()

if __name__ == "__main__":
    asyncio.run(main())

# Can I use ChatGPT for my assignment?
# What happens if I get sick during an exam?
# How is the AI & Machine Learning course graded?