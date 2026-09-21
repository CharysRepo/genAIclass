from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.chat_agent import chat_agent
from utils.utils import convert_pydanticai_to_openai

chat_router = APIRouter(prefix="/chat", tags=["chat"])

history_map = {}

class ChatRequest(BaseModel):
    content: str

@chat_router.post("/{user_id}")
async def chat_endpoint(user_id: str, request: ChatRequest):
    try:
        history = history_map.get(user_id, [])
        
        result = await chat_agent.run(request.content, message_history=history)
        
        history_map[user_id] = result.all_messages()
        
        return {"content": result.output}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@chat_router.get("/history/{user_id}")
async def get_history_endpoint(user_id: str):
    try:
        history = history_map.get(user_id, [])
        history_openai = convert_pydanticai_to_openai(history)
        return {"history": history_openai}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
