from typing import Annotated, Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import logfire
from dotenv import load_dotenv

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

class TicketCategory(BaseModel):
    category: Annotated[
        Literal["general", "order", "billing"],
        Field(description="The category of the ticket")
    ]

agent = Agent(
    #'ollama:llama3-groq-tool-use:8b',
    'groq:llama-3.3-70b-versatile',
    instructions="Classify the following message into a category",
    output_type=TicketCategory,
)

response = agent.run_sync("I would like to place an order.")
print(response.output.category)
print(response.usage())

response = agent.run_sync("I have a question about my billing.")
print(response.output.category)
print(response.usage())
