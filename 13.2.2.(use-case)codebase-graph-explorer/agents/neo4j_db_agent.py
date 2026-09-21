from session_persistence.persistence import SessionPersistence, RedisStorageService
from guard_rails.prompt_injection_guard_rail import PromptInjectionGuardrail
from context_mgmt.compaction import Compaction
from error_logger.logger import ErrorLogger
from pydantic_ai import Agent
from capabilities.neo4j_db_capability import Neo4jCapability, DBDeps
from config.config_reader import settings

db_agent = Agent(
    settings.CHAT_MODEL,
    #'groq:llama-3.3-70b-versatile',
    #"ollama:glm-4.7-flash:q4_K_M",
    deps_type=DBDeps,
    capabilities=[
        PromptInjectionGuardrail(),
        SessionPersistence(store=RedisStorageService(host="localhost", port=6379)),
        Compaction(),
        ErrorLogger(),
        Neo4jCapability()
    ]
)