import re
from typing import Any
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from dataclasses import dataclass

@dataclass
class SecretLeakageGuardrail(AbstractCapability[Any]):

    async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
        SECRET_PATTERNS = [
            re.compile(r'sk-[a-zA-Z0-9]{20,}'),  # OpenAI keys
            re.compile(r'ghp_[a-zA-Z0-9]{36,}'),  # GitHub PATs
            re.compile(r'AKIA[A-Z0-9]{16}'),  # AWS access keys
            re.compile(r'xoxb-[a-zA-Z0-9\-]+'),  # Slack bot tokens
            re.compile(r'Bearer\s+[a-zA-Z0-9\-._~+/]+=*'),  # Bearer tokens
        ]        
        output_str = str(result.output)
        if any(pattern.search(output_str) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Secret leakage detected in output: '{output_str}'")
        return result
