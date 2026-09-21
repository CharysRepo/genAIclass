from typing import Any
from dotenv import load_dotenv
import logfire
from pydantic_ai import Agent, AgentToolset, FunctionToolset,  RunContext
from pydantic_ai.capabilities import AbstractCapability
from dataclasses import dataclass
from pydantic_ai._instructions import AgentInstructions

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

@dataclass
class WeatherCapability(AbstractCapability[Any]):

    def get_instructions(self) -> AgentInstructions[Any] | None:
        """Return dynamic instructions that include stored memories."""
        return "Use these tools for current temperature and wind forecasts."

    def get_temperature(self, ctx:RunContext[None], city: str) -> str:
        """Get the current temperature for a city."""
        return f"The temperature in {city} is 22°C."

    def get_wind_speed(self, ctx:RunContext[None], city: str) -> str:
        """Get the current wind speed for a city in km/h."""
        return f"The wind speed in {city} is 15 km/h."
    
    def get_toolset(self) -> AgentToolset[Any] | None:
        """Build and return the toolset containing all file system tools."""
        toolset: FunctionToolset[Any] = FunctionToolset()
        toolset.add_function(self.get_temperature, name='get_temperature')
        toolset.add_function(self.get_wind_speed, name='get_wind_speed')
        return toolset

agent = Agent(
    'ollama:qwen3.5:9b',
    capabilities=[WeatherCapability()],
    instructions="You are a helpful assistant."
)


async def main():
    result = await agent.run("What's the temperature in london?")
    print(f"Output: {result.output}")
    result = await agent.run("What's the wind speed in london?")
    print(f"Output: {result.output}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())