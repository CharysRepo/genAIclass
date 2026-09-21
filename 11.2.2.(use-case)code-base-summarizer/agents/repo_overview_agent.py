from pydantic_ai import Agent
from config.config_reader import settings

repo_overview_agent = Agent(
    settings.SUMMARY_MODEL,
    instructions="""
      You are an expert technical writer. Draft a high-level overview
      for the root of a README.md.
    """
)
