from dotenv import load_dotenv
import logfire
from pydantic_ai import Agent, FunctionToolset, RunContext

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

weather_toolset = FunctionToolset()

@weather_toolset.instructions
def math_instructions(ctx: RunContext[str]) -> str:
    return 'Use these tools for current temperature and wind forecasts.'


@weather_toolset.tool
def get_temperature(ctx:RunContext[None],city: str) -> str:
    """Get the current temperature for a city."""
    return f"The temperature in {city} is 22°C."

@weather_toolset.tool
def get_wind_speed(ctx:RunContext[None], city: str) -> str:
    """Get the current wind speed for a city in km/h."""
    return f"The wind speed in {city} is 15 km/h."

agent = Agent(
    'ollama:qwen3.5:9b',
    toolsets=[weather_toolset],
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