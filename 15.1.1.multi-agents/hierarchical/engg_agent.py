from pydantic import BaseModel, Field
from pydantic_ai import Agent

class EscalationTicket(BaseModel):
    technical_severity: str = Field(description="Severity ranking: Low, Medium, High, Critical.")
    system_component: str = Field(description="The backend service affected (e.g., Auth, Database, Payment Gateway).")
    internal_notes: str = Field(description="Technical summary generated for the engineering team.")

# Handles complex technical issues requiring engineering logs.
engineering_agent = Agent(
    'ollama:qwen3.5:9b',
    output_type=EscalationTicket,
    system_prompt=(
        "You are a Tier-2 Technical Escalation Agent. Diagnose technical bugs, "
        "determine system severity, and draft an internal ticket for engineering teams."
    )
)
