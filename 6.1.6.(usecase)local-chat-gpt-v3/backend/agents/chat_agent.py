from pydantic_ai import Agent
from config.config_reader import settings

chat_agent = Agent(
    settings.MODEL, 
    instructions="You are helpful assistant."
)