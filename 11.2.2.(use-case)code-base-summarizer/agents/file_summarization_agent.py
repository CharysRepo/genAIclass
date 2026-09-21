from pydantic_ai import Agent
from config.config_reader import settings

file_summarization_agent = Agent(
    settings.SUMMARY_MODEL,
    instructions="""
      You are a senior software engineer creating a concise summary of a
      source file for a project's README.md.
    """
)
