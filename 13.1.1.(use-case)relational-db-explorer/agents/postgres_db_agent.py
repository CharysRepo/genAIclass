from session_persistence.persistence import SessionPersistence, RedisStorageService
from guard_rails.prompt_injection_guard_rail import PromptInjectionGuardrail
from context_mgmt.compaction import Compaction
from error_logger.logger import ErrorLogger
from pydantic_ai import Agent
from capabilities.postgres_db_capability import PostgresCapability, DBDeps
from config.config_reader import settings

db_agent = Agent(
    settings.CHAT_MODEL,
    deps_type=DBDeps,
    capabilities=[
        PromptInjectionGuardrail(),
        SessionPersistence(store=RedisStorageService(host=settings.REDIS_HOST, port=settings.REDIS_PORT)),
        Compaction(),
        ErrorLogger(),
        PostgresCapability()
    ]
)