import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelMessagesTypeAdapter
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are helpful assistant."
)

history_path = Path(__file__).parent.parent.resolve() / "data" / "history.json"
with open(history_path, "rb") as f:
    history = json.load(f)

history = ModelMessagesTypeAdapter.validate_python(history)
result = agent.run_sync("Tell me a different motto compared to past.", message_history=history)
print(result.output)
print(result.usage())

