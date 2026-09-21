from typing import Any
from pydantic_ai import  RunContext, ToolDefinition, FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from dataclasses import dataclass
from pydantic_ai import AgentToolset
from pydantic_ai._instructions import AgentInstructions

@dataclass
class ToolGuard(AbstractCapability[Any]):
    
    async def prepare_tools(self, ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        if ctx.deps.role == "user":
            return [tool_def for tool_def in tool_defs if tool_def.name == "get_temperature"]
        elif ctx.deps.role == "admin":
            return tool_defs
        else:
            return None
        
    def get_instructions(self) -> AgentInstructions[Any] | None:
        """Return dynamic instructions that include stored memories."""
        return "Use these tools for current temperature and wind forecasts."

    def get_temperature(self, ctx:RunContext[None], city: str) -> str:
        """Get the current temperature for a city."""
        return f"The temperature in {city} is 22°C."

    def get_wind_speed(self, ctx:RunContext[None], city: str) -> str:
        """Get the current wind speed for a city in km/h."""
        return f"The wind speed in {city} is 15 km/h."
        
    def get_toolset(self) -> AgentToolset[Any] | None:
        """Build and return the toolset containing all file system tools."""
        toolset: FunctionToolset[Any] = FunctionToolset()
        toolset.add_function(self.get_temperature, name='get_temperature')
        toolset.add_function(self.get_wind_speed, name='get_wind_speed')
        return toolset