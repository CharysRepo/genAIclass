from typing import Annotated
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import logfire
from dotenv import load_dotenv

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

class Step(BaseModel):
    explanation: Annotated[str, Field(description="The step-by-step logic used to solve this part of the problem.")]
    output: Annotated[str, Field(description="The mathematical result or expression derived in this step.")]

class Reasoning(BaseModel):
    steps: Annotated[list[Step], Field(description="A sequence of logical steps leading to the solution.")]
    final_answer: Annotated[str, Field(description="The simplified final result of the mathematical problem.")]



agent = Agent(
    #'ollama:llama3-groq-tool-use:8b',
    'groq:llama-3.3-70b-versatile',
    instructions="You are a helpful assistant to extract structure.",
    output_type=Reasoning,
)

user_prompt = "how can I solve 8x + 7 = -23"
response = agent.run_sync(user_prompt)
print(response.output)
print(response.usage())

