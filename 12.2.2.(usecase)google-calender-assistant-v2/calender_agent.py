from pydantic_ai import Agent
from calender_capability import CalenderCapability
from session_persistence.persistence import SessionPersistence, RedisStorageService
from guard_rails.prompt_injection_guard_rail import PromptInjectionGuardrail
from context_mgmt.compaction import Compaction
from error_logger.logger import ErrorLogger
from calender_service import CalenderService
from dataclasses import dataclass

@dataclass
class Deps:
    calender_service: CalenderService

calender_agent = Agent(
    "ollama:qwen3.5:9b",
    #"ollama:glm-4.7-flash:q4_K_M",
    deps_type=Deps,
    capabilities=[
        PromptInjectionGuardrail(),
        SessionPersistence(session_id="session1", store=RedisStorageService(host="localhost", port=6379)),
        Compaction(),
        CalenderCapability(),
        ErrorLogger()
    ]    
)
