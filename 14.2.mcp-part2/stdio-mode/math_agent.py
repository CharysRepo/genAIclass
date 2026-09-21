from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from dotenv import load_dotenv
load_dotenv(override=True)
from error_logger.logger import ErrorLogger
from fastmcp.client.transports import StdioTransport
from pydantic_ai.capabilities.toolset import Toolset

stdio_transport = StdioTransport(
    command= "C:/Users/Algorithmica/anaconda3/envs/genai-2026/python.exe",
    args=["F:/GitHub/algo-genai-2026/14.1.mcp/stdio-mode/mcp_server.py", "--transport", "stdio"]
)

mcp_toolset = MCPToolset(stdio_transport, cache_tools=True, include_instructions=True)
#mcp_toolset = MCPToolset("http://localhost:8000/mcp", cache_tools=True, include_instructions=True)

math_agent = Agent(
        'ollama:qwen3.5:9b', 
        capabilities=[
            Toolset(mcp_toolset),
            ErrorLogger()
        ]
    )
