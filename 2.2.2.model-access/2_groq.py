from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv(override=True)

agent = Agent(
    'groq:llama-3.3-70b-versatile',
    instructions="You are a helpful assistant.",
)

response = agent.run_sync("Write a haiku about recursion in programming.")
print(response.output)
print(response.usage())

response = agent.run_sync("What is recursion in programming.")
print(response.output)
print(response.usage())