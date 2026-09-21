from typing import Any
from pydantic import ValidationError
from pydantic_ai import AgentRunResult, ModelResponse, ModelRetry, RunContext, ModelRequestContext, TextPart
from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs
from dataclasses import dataclass

@dataclass
class ErrorLogger(AbstractCapability[Any]):
    """Logs all errors that occur during agent runs."""

    async def on_model_request_error(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        print(f'Model error: {error}')
        return ModelResponse(parts=[TextPart(content='Service temporarily unavailable.')])
    
    async def on_tool_validation_error(
        self, ctx: RunContext[Any], *, call: Any, tool_def: Any, args: dict[str, Any], error: ValidationError | ModelRetry
    ) -> ValidatedToolArgs:
        print(f"Validation failed for tool '{call.tool_name}': {error}")
        raise error
    
    async def on_tool_execute_error(
        self, ctx: RunContext[Any], *, call: Any, tool_def: Any, args: dict[str, Any], error: Exception
    ) -> Any:
        print(f'Tool {call.tool_name} failed: {error}')
        raise error 
    
    async def on_run_error(self, ctx: RunContext[Any], error: BaseException) -> AgentRunResult[Any]:
        print(f'Run error: {error}')
        return AgentRunResult(output='An unexpected error occurred.')

