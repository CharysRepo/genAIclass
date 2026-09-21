from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv(override=True)

agent = Agent(
    'ollama:tinyllama:latest',
    instructions="You are a helpful assistant",
)

#response = agent.run_sync("can you translate any language?")
#print(response.output)
#print(response.usage())

response = agent.run_sync("what is recursion in programming.")
print(response.output)
print(response.usage())