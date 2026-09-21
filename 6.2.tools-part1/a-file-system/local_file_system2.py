import asyncio
import os
from pathlib import Path
from openai import BaseModel
from pydantic_ai import Agent
from dotenv import load_dotenv
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

class FileOutput(BaseModel):
    files: list[str]


fs_agent = Agent(
    #'groq:llama-3.3-70b-versatile',
    'github:openai/gpt-4o',
    #"ollama:llama3-groq-tool-use:8b",
    #"ollama:qwen3.5:9b",
    output_type= FileOutput,
    instructions="You are a helpful assistant. Use the tools provided to find requested infomration."
)

@fs_agent.tool_plain
def get_cwd() -> str:
    """Returns the current working directory."""
    return os.getcwd()

@fs_agent.tool_plain
def list_all_files(dir: str) -> list[str]:
    """
    List all files in the input directory

    Args:
        dir (str): The directory to list files from

    Returns:
        (list[str]): A list of all file paths in the input directory
    """
    logfire.info(f"listing files of {dir}")
    files = [str(path) for path in Path(dir).glob("**/*")]
    return files

async def main(user_prompt):
    response = await fs_agent.run(user_prompt)
    print(response.output)

if __name__ == "__main__":
    user_prompt = "What are the files under current working directory"
    asyncio.run(main(user_prompt))

# "What are the files in the directory: F:\GitHub\genai-2026\hooks"
# "What is the current working directory"