import asyncio
from typing import Annotated, Literal, List, Optional
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass, field
from datetime import date
from session_persistence.persistence import SessionPersistence, RedisStorageService
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire

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
class BookingState:
    collected_data: dict = field(default_factory=dict)
    field_order: List[str] = field(default_factory=lambda: [
        "source_city", "target_city", "booking_date", "trip_type", "adults", "children"
    ])

    def get_next_field(self) -> Optional[str]:
        for f in self.field_order:
            if f not in self.collected_data:
                return f
        return None

agent = Agent(
    "ollama:qwen3.5:9b",
    deps_type=BookingState,
    capabilities=[SessionPersistence(RedisStorageService(host="localhost", port=6379))],
    output_type=str | FlightBooking,
    instructions="You are a strict data collector. MANDATORY: You must call 'update_booking' to save the current field before asking for the next one. "
)

@agent.tool
def update_booking(ctx: RunContext[BookingState], field_name: str, value: str) -> str:
    """Save a piece of flight information to the state."""
    if field_name not in ctx.deps.field_order:
        return f"Error: {field_name} is not a valid field."
    
    ctx.deps.collected_data[field_name] = value
    tmp = f"Successfully saved {field_name}."
    if ctx.deps.get_next_field():
        tmp += f" Now ask for the next field {ctx.deps.get_next_field()} in the sequence."
    else:
        tmp += " All fields collected. you transform the source_city and target_city to their IATA code and then output the final FlightBooking object."
    return tmp

async def main():
    state = BookingState()

    print("AI: Hello! Welcome to the flight booking assistant. "
        "To help you find the best flights, please provide your **source** and **destination** cities"
    )

    
    while True:
        user_input = input("You: ")
        
        result = await agent.run(user_input, deps=state)        
        print(state)
        
        if isinstance(result.output, FlightBooking):
            print(f"\n✅ Booking Complete!\n{result.output.model_dump_json(indent=2)}")
            break
        else:
            print(f"AI: {result.output}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())