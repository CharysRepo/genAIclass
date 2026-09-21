from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext
from dataclasses import dataclass, field
from uuid import uuid4
from typing import Any
import redis.asyncio as redis
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

class RedisStorageService:
    def __init__(self, host: str, port: int):
        self.redis_client = redis.Redis(host=host, port=port, db=0, decode_responses=True)

    async def load(self, session_id: str) -> list[ModelMessage]:
        try:
            existing_history_json = await self.redis_client.get(session_id)
            return ModelMessagesTypeAdapter.validate_json(existing_history_json) if existing_history_json else []
        except Exception as e:
            raise e

    async def save(self, session_id: str, all_messages: list[ModelMessage]) -> None:
        try:
            await self.redis_client.set(session_id, ModelMessagesTypeAdapter.dump_json(all_messages).decode())
        except Exception as e:
            raise e

    async def delete(self, session_id: str) -> None:
        try:
            await self.redis_client.delete(session_id)
        except Exception as e:
            raise e
        
@dataclass
class SessionPersistence(AbstractCapability[Any]):
    """Capability for saving and restoring conversation state across agent runs.

    On run start, loads any previously saved messages for the session and
    prepends them to the conversation. On run end, saves the full message
    history back to the store.
    """
    store: RedisStorageService
    session_id: str = field(default_factory=lambda: str(uuid4()))

    async def before_run(
        self,
        ctx: RunContext[Any],
    ) -> None:
        
        existing = await self.store.load(self.session_id)
        if existing:
            ctx.messages[:0] = existing

    async def after_run(
        self,
        ctx: RunContext[Any],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        
        await self.store.save(self.session_id, result.all_messages())
        return result