import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire
from agents.postgres_db_agent import db_agent
from capabilities.postgres_db_capability import DBDeps
from services.postgres_db_service import PostgresDBService

logfire.configure()
logfire.instrument_pydantic_ai()

async def main():
    async with PostgresDBService() as db_service:
        deps = DBDeps(db_service)

        print("Postgres DB Agent is ready! (Type 'exit' to quit)")        
        while True:
            user_input = input("\nUser: ")
            if user_input.lower() in ("exit", "quit"):
                break
            result = await db_agent.run(
                user_input, 
                deps=deps
            )
            print(result.output)

if __name__ == "__main__":
    asyncio.run(main())

# What are the top 5 states by number of customers?
# What are the most popular payment methods?
# Which sellers have most orders?
# Whats the average review score?
# How many orders were delivered in 2017?
# Can you tell me the yearly breakdown of total revenue generated?
# Find the top 10 customers who spent the most money across all their orders.