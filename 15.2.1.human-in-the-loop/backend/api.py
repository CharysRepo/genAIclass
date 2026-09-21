from pydantic import BaseModel
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from test_agent import agent
from fastapi import APIRouter

class ChatRequest(BaseModel):
    message: str
    approvals: dict[str, bool] = {}

chat_router = APIRouter(prefix="/chat", tags=["chat"])

@chat_router.post("/greet")
async def greet(request: ChatRequest):
    print(request)
    if request.approvals:
        deferred_results = DeferredToolResults(approvals=request.approvals)
        print(deferred_results)
        result = await agent.run(
            deferred_tool_results=deferred_results
        )
    else:
        result = await agent.run(request.message)
    print(result.output)
    print(result.all_messages())

    if isinstance(result.output, DeferredToolRequests):
        return {
            "status": "requires_approval",
            "approvals": result.output.approvals,
            "output": None
        }
    else:
        return {
            "status": "complete",
            "approvals": [],
            "output": result.output
        }