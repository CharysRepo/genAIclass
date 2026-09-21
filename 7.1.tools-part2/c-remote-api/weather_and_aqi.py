# https://open-meteo.com/en/docs/air-quality-api (current air quality)
# https://open-meteo.com/en/docs (current weather)
import asyncio
import httpx
from datetime import datetime
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

weather_codes = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Drizzle: Light intensity",
    53: "Drizzle: Moderate intensity",
    55: "Drizzle: Dense intensity",
    56: "Freezing Drizzle: Light intensity",
    57: "Freezing Drizzle: Dense intensity",
    61: "Rain: Slight intensity",
    63: "Rain: Moderate intensity",
    65: "Rain: Heavy intensity",
    66: "Freezing Rain: Light intensity",
    67: "Freezing Rain: Heavy intensity",
    71: "Snow fall: Slight intensity",
    73: "Snow fall: Moderate intensity",
    75: "Snow fall: Heavy intensity",
    77: "Snow grains",
    80: "Rain showers: Slight",
    81: "Rain showers: Moderate",
    82: "Rain showers: Violent",
    85: "Snow showers: Slight",
    86: "Snow showers: Heavy",
    95: "Thunderstorm: Slight or moderate",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}


@dataclass
class ResearchDeps:
    client: httpx.AsyncClient

weather_agent = Agent(
    #'groq:llama-3.3-70b-versatile',
    #'github:openai/gpt-4o',
    "ollama:llama3-groq-tool-use:8b",
    #"ollama:qwen3.5:9b",
    deps_type=ResearchDeps,
    instructions="You are a helpful assistant. Use the tools provided to find requested infomration. Donot use any rounding or truncation while answering."
)

@weather_agent.tool
async def get_weather(ctx: RunContext[ResearchDeps], lat: float, lon: float) -> str:
    """Get current weather for a specific latitude and longitude."""

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "weather_code"]
    }
    resp = await ctx.deps.client.get(url, params=params)
    data = resp.json().get("current", {})
    temperature_2m = data.get('temperature_2m')
    weather_code = data.get('weather_code')
    print(temperature_2m, weather_code, weather_codes[weather_code])
    return f"Temperature: {temperature_2m}°C, weathercode: {weather_codes[weather_code]}"

@weather_agent.tool
async def get_aqi(ctx: RunContext[ResearchDeps], lat: float, lon: float) -> str:
    """Get current air quality index for a specific latitude and longitude."""

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm10", "pm2_5"]
    }
    resp = await ctx.deps.client.get(url, params=params)
    data = resp.json().get("hourly", {})
    pm10_hourly = data.get('pm10')
    pm2_5_hourly = data.get('pm2_5')
    pm10_current_hour = pm10_hourly[datetime.now().hour]
    pm2_5_current_hour = pm2_5_hourly[datetime.now().hour]
    print(pm10_current_hour, pm2_5_current_hour)
    return f"pm10: {pm10_current_hour}, pm2_5: {pm2_5_current_hour}"

async def main(user_prompt):
    async with httpx.AsyncClient() as http_client:
        deps = ResearchDeps(client=http_client)        
        result = await weather_agent.run(user_prompt, deps=deps)
        print(result.output)

if __name__ == "__main__":
    user_prompt = "What are the air quality index and weather of Hyderabad (Lat: 17.38, Lon: 78.48)"
    asyncio.run(main(user_prompt))

# "What is the weather like in New York (Lat: 40.71, Lon: -74.00)"
# "What is the weather like in Hyderabad (Lat: 17.38, Lon: 78.48)"
# "What is the air quality index of Hyderabad (Lat: 17.38, Lon: 78.48)"
# "What is the air quality index of Delhi (Lat: 28.70, Lon: 77.10)"