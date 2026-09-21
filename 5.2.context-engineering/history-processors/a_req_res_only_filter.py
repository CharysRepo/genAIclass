import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelMessage, ModelMessagesTypeAdapter, ModelRequest, ModelResponse
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

def request_only_filter(messages: list[ModelMessage]) -> list[ModelMessage]:
    print("request only filter")
    return [msg for msg in messages if isinstance(msg, ModelRequest)]

def response_only_filter1(messages: list[ModelMessage]) -> list[ModelMessage]:
    print("response only filter")
    return[msg for msg in messages if isinstance(msg, ModelResponse)]

def response_only_filter2(messages: list[ModelMessage]) -> list[ModelMessage]:
    print("response only filter")
    temp = [msg for msg in messages[:-1] if isinstance(msg, ModelResponse)]
    temp.append(messages[-1])
    return temp


agent = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are helpful assistant.",
    history_processors=[response_only_filter2]
)

history_path = Path(__file__).parent.resolve() / "data" / "history.json"
with open(history_path, "rb") as f:
    history = json.load(f)

history = ModelMessagesTypeAdapter.validate_python(history)
result = agent.run_sync("Tell me a different motto compared to past.", message_history=history)
print(result.output)
print(result.usage())