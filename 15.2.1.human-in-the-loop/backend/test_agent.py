from pydantic_ai import Agent, DeferredToolRequests, RunContext
from session_persistence.persistence import SessionPersistence, RedisStorageService
from error_logger.logger import ErrorLogger

agent = Agent(
    'ollama:qwen3.5:9b',
    output_type=str | DeferredToolRequests,
    capabilities=[
        SessionPersistence(store=RedisStorageService(host="localhost", port=6379)), 
        ErrorLogger()
    ]
)

@agent.tool(requires_approval=True)
async def delete_file(ctx: RunContext[None], file_path: str) -> str:
    """delete a file."""
    return f"Action executed with detail: {file_path}"