from dotenv import load_dotenv
from pydantic_ai import Agent
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent1 = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are helpful assistant."
)

agent2 = Agent(
    'groq:llama-3.3-70b-versatile',
    instructions="You are helpful assistant."
)

result = agent1.run_sync("tell me a joke, in short", message_history=[])
print(result.output)

history = result.all_messages()
result = agent2.run_sync("explain", message_history=history)
print(result.output)
