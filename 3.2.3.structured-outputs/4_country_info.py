from typing import Annotated
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import logfire
from dotenv import load_dotenv

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

Flag = Annotated[bool, Field(description="True if the country has a significant river or coastline")]

class CountryInfo(BaseModel):
    continent_name: Annotated[str, Field(description="Name of the continent (e.g., 'Europe', 'Asia')")]
    country_name: Annotated[str, Field(description="Official name of the country")]
    capital_name: Annotated[str, Field(description="Primary capital city of the country")]
    has_river: Flag
    has_sea: Flag
    weather: Annotated[str, Field(description="Brief summary of the typical climate")]


agent = Agent(
    #'ollama:llama3-groq-tool-use:8b',
    'groq:llama-3.3-70b-versatile',
    instructions="You are a helpful assistant.",
    output_type=CountryInfo,
)

response = agent.run_sync("tell me about India")
print(response.output)
print(response.usage())

response = agent.run_sync("tell me about Australia")
print(response.output)
print(response.usage())
