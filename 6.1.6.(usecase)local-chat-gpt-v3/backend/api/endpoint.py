from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from agents.chat_agent import chat_agent
from services.conversation import ConversationService
from utils.utils import convert_pydanticai_to_openai

chat_router = APIRouter(prefix="/chat", tags=["chat"])   
conversation_service = ConversationService()

@chat_router.websocket("/ws/{user_id}")
async def chat_websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()    
    history = await conversation_service.fetch_history(user_id)
    print(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            user_input = data.get("content", "")

            await websocket.send_json({"type": "start"})

            async with chat_agent.run_stream(user_input, message_history=history) as result:
                async for message in result.stream_text(delta=True):
                    await websocket.send_json({"type": "token", "content": message})

            history.extend(result.new_messages())
            await conversation_service.update_history(user_id, result.all_messages())
            await websocket.send_json({"type": "end"})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "content": str(e)})

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
