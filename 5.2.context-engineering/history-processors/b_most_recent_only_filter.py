from dataclasses import dataclass
import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelMessage, ModelMessagesTypeAdapter, RunContext
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

def most_recent_only_filter(ctx:RunContext[Deps], messages: list[ModelMessage]) -> list[ModelMessage]:
    print("most recent only filter")
    return  messages[-ctx.deps.num_messages_to_keep:]

agent = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are helpful assistant.",
    history_processors=[most_recent_only_filter]
)

history_path = Path(__file__).parent.resolve() / "data" / "history.json"
with open(history_path, "rb") as f:
    history = json.load(f)

@dataclass
class Deps:
    num_messages_to_keep: int

deps = Deps(num_messages_to_keep=3)
history = ModelMessagesTypeAdapter.validate_python(history)
result = agent.run_sync("Tell me a different motto compared to past.", message_history=history, deps=deps)
print(result.output)
print(result.usage())