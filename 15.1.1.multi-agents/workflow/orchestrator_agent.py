from typing import Literal
from pydantic_ai import Agent
from pydantic import BaseModel, Field

class RoutingDecision(BaseModel):
    """Router outputs only the destination choice to decouple execution."""
    target_agent: Literal["faq", "engineering"] = Field(
        description="Select 'faq' for policy/returns or 'engineering' for database/code errors."
    )
    justification: str = Field(description="Brief reason for routing choice.")

orchestrator_agent = Agent(
    'ollama:qwen3.5:9b',
    output_type=RoutingDecision,
    system_prompt=(
        "You are a triage router. Categorize incoming tickets.\n"
        "1. For routine topics (returns, refunds, tracking), select 'faq'.\n"
        "2. For exceptions, database lockups, or server crashes, select 'engineering'."
    )
)

