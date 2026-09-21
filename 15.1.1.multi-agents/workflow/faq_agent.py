from pydantic import BaseModel, Field
from pydantic_ai import Agent

class AutoResolution(BaseModel):
    policy_applied: str = Field(description="The company policy used to resolve the issue.")
    resolution_steps: str = Field(description="Step-by-step instructions provided to the customer.")
    is_resolved: bool = Field(default=True, description="Always True for auto-resolved items.")

# Handles standard, policy-driven self-service requests.
faq_agent = Agent(
    'ollama:qwen3.5:9b',
    output_type=AutoResolution,
    system_prompt=(
        "You are an Automated Tier-1 Support Agent. Resolve standard requests instantly "
        "using company policy. Provide clear, direct steps to the user."
    )
)