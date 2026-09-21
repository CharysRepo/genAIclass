from dotenv import load_dotenv
from pydantic_ai import Agent
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

def print_result(result):
    print("\n\n### Output")
    print(f"\n {result.output}")
    print("\n\n### New messages")
    messages = result.new_messages()
    for idx, message in enumerate(messages):
        print(f"\nMessage #{idx + 1}: {type(message).__name__}")
        print(message)
    print("\n\n### All messages")
    messages = result.all_messages()
    for idx, message in enumerate(messages):
        print(f"\nMessage #{idx + 1}: {type(message).__name__}")
        print(message)

agent = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are helpful assistant."
)

result = agent.run_sync("i am thimmareddy and i am a researcher of AI and Consciousness")
print_result(result)

result = agent.run_sync("what is my name?")
print_result(result)
