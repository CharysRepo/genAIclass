from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated
from pydantic_ai import Agent
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

class AddressInfo(BaseModel):
    first_name: Annotated[str, Field(min_length=1, max_length=50)]
    last_name: Annotated[str, Field(min_length=1, max_length=50)]
    street: Annotated[str, Field(description="Street name or primary address line")]
    house_number: Annotated[str, Field(pattern=r"^[0-9a-zA-Z\s-]+$")]
    postal_code: Annotated[str, Field(min_length=3, max_length=10)]
    city: Annotated[str, Field(examples=["Berlin"])]
    state: Annotated[str, Field(description="State or Province")]
    country: Annotated[str, Field(min_length=2, max_length=2, description="ISO 3166-1 alpha-2 code")]

agent = Agent(
    #'ollama:llama3-groq-tool-use:8b',
    'groq:llama-3.3-70b-versatile',
    instructions="You are a helpful assistant.",
    output_type=AddressInfo,
)

user_prompt = "Sherlock Holmes lives in the United Kingdom. His residence is in at 221B Baker Street, London, NW1 6XE."
response = agent.run_sync(user_prompt)
print(response.output)
print(response.usage())