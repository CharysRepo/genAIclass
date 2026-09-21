from pydantic_ai import Agent
from config.config_reader import settings

summarizer_agent = Agent(
    settings.CHAT_MODEL,
    instructions="""
      Your role is to create a concise, factual summary of a document.
    """
)

