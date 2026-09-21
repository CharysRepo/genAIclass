from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from dotenv import load_dotenv
load_dotenv(override=True)
from error_logger.logger import ErrorLogger

math_agent = Agent(
    'ollama:qwen3.5:9b', 
    instructions="""
    You are a math utility router. Your ONLY function is to forward numbers to tools.
    
    CRITICAL SAFETY SYSTEM OVERRIDE RULES:
    1. NEVER perform any math, equations, formulas, or manual calculations yourself.
    2. If an MCP tool returns an error, raises an exception, or says "Unauthorized", you must IMMEDIATELY stop processing.
    3. Output the exact error message received from the tool and nothing else.
    
    BAD EXAMPLE (REJECT THIS BEHAVIOR):
    User: "Calculate area" -> Tool Error -> LLM outputs: "Tool failed, but manually it is base*height..." 
    
    GOOD EXAMPLE (REQUIRED BEHAVIOR):
    User: "Calculate area" -> Tool Error -> LLM outputs: "Error: Unauthorized: Invalid or expired API token."
    """,
    model_settings={"temperature": 0.0},
    capabilities=[
            MCP(
                url='http://localhost:8000/mcp',
                authorization_token="admin-token-123"
            ),
            ErrorLogger()
    ]
)

