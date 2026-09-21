from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from dotenv import load_dotenv
load_dotenv(override=True)
from error_logger.logger import ErrorLogger

math_agent = Agent(
        'ollama:qwen3.5:9b', 
        capabilities=[
            MCP(
                url='http://localhost:8000/mcp'
            ),
            ErrorLogger()
        ]
    )

# mcp/sse