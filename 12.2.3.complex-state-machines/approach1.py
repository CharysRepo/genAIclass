import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)
from datetime import date
from typing import Literal, List
from pydantic import BaseModel, Field
from session_persistence.persistence import SessionPersistence, RedisStorageService
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass, field
import logfire
from typing import Annotated, Literal
from pydantic import BaseModel, Field
logfire.configure()
logfire.instrument_pydantic_ai()

class FlightBooking(BaseModel):
    source_city: Annotated[
        str, 
        Field(description="Source city name or 3-letter IATA code (e.g., JFK)", examples=["JFK", "London"])
    ]
    target_city: Annotated[
        str, 
        Field(description="Target city name or 3-letter IATA code (e.g., LAX)", examples=["LAX", "Paris"])
    ]
    booking_date: Annotated[
        date, 
        Field(description="Booking date in ISO 8601 format (YYYY-MM-DD)", examples=["2024-12-25"])
    ]
    trip_type: Literal['oneway', 'twoway']
    adults: Annotated[int, Field(ge=1, le=10, description="Number of adult passengers (age 12+)")]
    children: Annotated[int, Field(ge=0, le=10, description="Number of child passengers (age 2-11)")]


@dataclass
class Deps:
    field_order: List[str] = field(default_factory=lambda: [
        "source_city", "target_city", "booking_date", "trip_type", "adults", "children"
    ])

agent = Agent(
    "ollama:qwen3.5:9b",
    deps_type=Deps,
    capabilities=[SessionPersistence(RedisStorageService(host="localhost", port=6379))],
    output_type=str | FlightBooking, 
)

@agent.instructions
def sequential_enforcer(ctx: RunContext[Deps]) -> str:
    return (
        f"You are a strict data collector. You MUST collect info in this order: "
        f"{', '.join(ctx.deps.field_order)}. "
    )

async def main():
    deps = Deps()

    print(
        "AI: Hello! Welcome to the flight booking assistant. "
        "To help you find the best flights, please provide your **source** and **destination** cities, "
        "your **travel date** (YYYY-MM-DD), and how many **adults** and **children** will be traveling."
    )
    
    while True:
        user_input = input("You: ")
        
        result = await agent.run(user_input, deps=deps)        
        if isinstance(result.output, FlightBooking):
            print(f"\n✅ Booking Complete!\n{result.output.model_dump_json(indent=2)}")
            break
        else:
            print(f"AI: {result.output}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())