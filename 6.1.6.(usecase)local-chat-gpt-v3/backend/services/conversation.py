import redis.asyncio as redis
from pydantic_ai import ModelMessage, ModelMessagesTypeAdapter
from config.config_reader import settings

class ConversationService:
    def __init__(self):
        self.redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, decode_responses=True)

    async def fetch_history(self, user_id:str) -> list[ModelMessage]:
        try:
            existing_history_json = await self.redis_client.get(f"chat_history:{user_id}")
            if existing_history_json:
                history = ModelMessagesTypeAdapter.validate_json(existing_history_json)
            else:
                history = []
            return history
        except Exception as e:
            raise e
    
    async def update_history(self, user_id:str, all_messages:list[ModelMessage]) -> None:
        try:
            await self.redis_client.set(f"chat_history:{user_id}", ModelMessagesTypeAdapter.dump_json(all_messages).decode())
        except Exception as e:
            raise e