from pydantic_ai import Agent, ModelMessage, RunContext
from dataclasses import dataclass
from config.config_reader import settings

chat_agent = Agent(
    settings.MODEL, 
    instructions="You are helpful assistant."
)