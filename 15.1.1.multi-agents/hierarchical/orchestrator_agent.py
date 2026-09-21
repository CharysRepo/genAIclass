from typing import Union
from faq_agent import AutoResolution, faq_agent
from engg_agent import EscalationTicket, engineering_agent
from pydantic_ai import Agent, RunContext

orchestrator_agent = Agent(
    'ollama:qwen3.5:9b',
    output_type=Union[AutoResolution, EscalationTicket],
    instructions="You are a Customer Support Assistant to analyze incoming user tickets.\n"

)

@orchestrator_agent.tool
async def route_to_faq_agent(ctx: RunContext[None], user_issue: str) -> AutoResolution:
    """Routes routine, policy-based customer issues to the automated FAQ subagent."""
    print(f"[Orchestrator]: User Issue: '{user_issue}'")
    print("[Orchestrator]: Ticket classified as Routine. Routing to FAQ Agent...")
    response = await faq_agent.run(user_issue, usage=ctx.usage)
    return response.output

@orchestrator_agent.tool
async def route_to_engineering_agent(ctx: RunContext[None], technical_issue: str) -> EscalationTicket:
    """Routes complex technical bugs and system failures to the Engineering subagent."""
    print(f"[Orchestrator]: Technical Issue: '{technical_issue}'")
    print("[Orchestrator]: Ticket classified as Complex Bug. Routing to Engineering Agent...")
    response = await engineering_agent.run(technical_issue, usage=ctx.usage)
    return response.output

