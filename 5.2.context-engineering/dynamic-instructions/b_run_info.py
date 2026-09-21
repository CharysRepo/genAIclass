from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    #'groq:llama-3.3-70b-versatile',
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are a helpful assistant.",
)

@agent.instructions
def dynamic_instructions(ctx:RunContext[UserProfile]):
    print("#### Information available in Runtime context")
    print(ctx.usage)
    print(ctx.deps)
    print(ctx.model)
    print(ctx.prompt)
    print(ctx.run_id)
    print(ctx.run_step)
    username = ctx.deps.name
    return f"Always include following username in the response. {username}"

@dataclass
class UserProfile:
    name: str

deps = UserProfile(name="john")
response = agent.run_sync("Tell me a joke in short", deps=deps)
print(response.output)
print(response.usage())

response = agent.run_sync("Tell me a joke in short", deps=deps)
print(response.output)
print(response.usage())
