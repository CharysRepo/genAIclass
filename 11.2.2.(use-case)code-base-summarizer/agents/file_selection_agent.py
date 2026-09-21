from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from config.config_reader import settings

class FileSelection(BaseModel):
    important_files: List[str] = Field(
        description="A list of file paths, ordered from most to least architecturally significant.",
        min_length=1,
        max_length=10
    )

    
file_selection_agent = Agent(
    settings.CHAT_MODEL,
    output_type=FileSelection,
    output_retries=3,
    instructions="""
      You are a senior software architect. Your task is to identify the
      most critical files for understanding a repository's architecture.
    """
)
