from dotenv import load_dotenv
from pydantic_ai import Agent
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    #'groq:llama-3.3-70b-versatile',
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are a helpful assistant.",
)

response = agent.run_sync("Write a haiku about recursion in programming.")
print(response.output)
print(response.usage())