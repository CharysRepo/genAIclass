from typing import Annotated
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import logfire
from dotenv import load_dotenv

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

class SubTask(BaseModel):
    task_id: Annotated[
        int, 
        Field(description="Unique identifier for the subtask", ge=1)
    ]
    description: Annotated[
        str, 
        Field(description="Detailed description of the action to take", min_length=1)
    ]

class Plan(BaseModel):
    tasks: Annotated[
        list[SubTask], 
        Field(description="List of ordered subtasks to achieve the goal")
    ]
    goal: Annotated[
        str, 
        Field(description="The main objective of the plan")
    ]


agent = Agent(
    #'ollama:llama3-groq-tool-use:8b',
    'groq:llama-3.3-70b-versatile',
    instructions="You are a helpful assistant to extract structure.",
    output_type=Plan,
)

user_prompt = "Find the population of France and multiply it by 2, then summarize."
response = agent.run_sync(user_prompt)
print(response.output)
print(response.usage())

