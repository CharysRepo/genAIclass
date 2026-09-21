from pydantic_ai import Agent
from config.config_reader import settings

reducer_agent = Agent(
    settings.CHAT_MODEL,
    instructions="you are a helpful assistant"
)

