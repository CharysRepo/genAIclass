import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire
from agents.neo4j_db_agent import db_agent
from services.neo4j_db_service import AsyncNeo4jDBService
from capabilities.neo4j_db_capability import DBDeps

logfire.configure()
logfire.instrument_pydantic_ai()

async def main():
    db_service =  AsyncNeo4jDBService()
    print("Neo4j DB Agent is ready! (Type 'exit' to quit)")
    deps=DBDeps(db_service=db_service)
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break
        result = await db_agent.run(user_input, deps=deps)
        print(result.output)

    await db_service.close()

if __name__ == "__main__":
    asyncio.run(main())

# find all the classes
# find all the methods of IngestionService
# give me the code of ingest_file
