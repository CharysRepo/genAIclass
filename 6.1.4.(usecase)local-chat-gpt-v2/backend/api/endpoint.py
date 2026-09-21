from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.chat_agent import chat_agent
from services.conversation import ConversationService
from utils.utils import convert_pydanticai_to_openai

chat_router = APIRouter(prefix="/chat", tags=["chat"])   
conversation_service = ConversationService()

class ChatRequest(BaseModel):
    content: str

@chat_router.post("/{user_id}")
async def chat_endpoint(user_id: str, request: ChatRequest):
    try:
        history = await conversation_service.fetch_history(user_id)

        result = await chat_agent.run(request.content, message_history=history)

        await conversation_service.update_history(user_id, result.all_messages())
        return {"content": result.output}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@chat_router.get("/history/{user_id}")
async def chat_endpoint(user_id: str):
    try:
        history = await conversation_service.fetch_history(user_id)
        history_openai = convert_pydanticai_to_openai(history) 
        #print(history_openai)    
        return {"history": history_openai}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
