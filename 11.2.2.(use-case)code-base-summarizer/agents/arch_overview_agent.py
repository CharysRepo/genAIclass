from pydantic_ai import Agent
from config.config_reader import settings

arch_overview_agent = Agent(
    settings.SUMMARY_MODEL,
    instructions="""
      You are a principal software architect. Use the provided file
      summaries (and raw code if present) to infer high-level design.
      Be precise and avoid guesswork.
    """
)
