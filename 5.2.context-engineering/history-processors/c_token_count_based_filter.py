from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelMessage, ModelMessagesTypeAdapter, RunContext
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

def token_usage_based_filter(ctx: RunContext[Deps], messages: list[ModelMessage]) -> list[ModelMessage]:
    print("token count based filter")
    current_tokens = ctx.deps.total_tokens
    print(current_tokens)
    token_limit = ctx.deps.token_limit
    num_messages_to_keep = ctx.deps.num_messages_to_keep

    if current_tokens > token_limit:
        return messages[-num_messages_to_keep:] 
    return messages


agent = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are helpful assistant.",
    history_processors=[token_usage_based_filter]
)

history_path = Path(__file__).parent.resolve() / "data" / "history.json"
with open(history_path, "rb") as f:
    history = json.load(f)

@dataclass
class Deps:
    total_tokens: int
    token_limit: int
    num_messages_to_keep: int

ntokens = 0
deps = Deps(token_limit=100, total_tokens=ntokens, num_messages_to_keep=3)
history = ModelMessagesTypeAdapter.validate_python(history)
result = agent.run_sync("Tell me a different motto compared to past.", message_history=history, deps=deps)
print(result.output)
print(result.usage())

ntokens += result.usage().input_tokens + result.usage().output_tokens
deps = Deps(token_limit=100, total_tokens=ntokens, num_messages_to_keep=3)
result = agent.run_sync("Tell me a different motto compared to past.", message_history=history, deps=deps)
print(result.output)
print(result.usage())