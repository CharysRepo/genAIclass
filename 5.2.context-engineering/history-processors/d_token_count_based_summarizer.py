from dataclasses import dataclass
import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelMessage, ModelMessagesTypeAdapter, RunContext
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

summarize_agent = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="Summarize this conversation, omitting small talk and unrelated topics."
)

def token_based_summarizer(ctx: RunContext[Deps], messages: list[ModelMessage]) -> list[ModelMessage]:
    print(ctx.deps.total_tokens)
    if ctx.deps.total_tokens > ctx.deps.token_limit:
        oldest_messages = messages[:-1]
        summary = summarize_agent.run_sync(message_history=oldest_messages)
        return summary.new_messages() + messages[-1:]

    return messages

agent = Agent(
    'ollama:llama3-groq-tool-use:8b',
    instructions="You are helpful assistant.",
    history_processors=[token_based_summarizer]
)

history_path = Path(__file__).parent.resolve() / "data" / "history.json"
with open(history_path, "rb") as f:
    history = json.load(f)

@dataclass
class Deps:
    total_tokens: int
    token_limit: int

ntokens = 0
deps = Deps(token_limit=100, total_tokens=ntokens)
history = ModelMessagesTypeAdapter.validate_python(history)
result = agent.run_sync("Tell me a different motto compared to past.", message_history=history, deps=deps)
print(result.output)
print(result.usage())

ntokens += result.usage().input_tokens + result.usage().output_tokens
deps = Deps(token_limit=100, total_tokens=ntokens)
result = agent.run_sync("Tell me a different motto compared to past.", message_history=history, deps=deps)
print(result.output)
print(result.usage())