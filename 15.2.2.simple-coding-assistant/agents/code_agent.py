from pydantic_ai import Agent
from capabilities.file_system import FileSystem
from capabilities.shell import Shell
from dataclasses import dataclass
from config.config_reader import settings

code_agent = Agent(
    settings.CHAT_MODEL,
    capabilities=[FileSystem(root_dir=settings.ROOT_DIR), Shell(cwd=settings.ROOT_DIR)],
    instructions="""
    You are a coding assistant to write code. You have access to a file system and a shell. 
    Use the file system to read and write code files, and use the shell to run commands and test your code. 
    Always write code in the file system and test it using the shell before providing it to the user. 
    Do not provide code directly to the user without writing it to a file first. 
    """
)

