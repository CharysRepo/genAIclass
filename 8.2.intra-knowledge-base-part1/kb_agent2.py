import asyncio
from pathlib import Path
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
import logfire
from dataclasses import dataclass

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()


@dataclass
class KnowledgeDeps:
    kb: str

kb_agent = Agent(
    #'groq:llama-3.3-70b-versatile',
     "ollama:glm-4.7-flash:q4_K_M",
    instructions="You are a helpful assistant. Always use the tool get_knowledge_base to get information before answering."
)

@kb_agent.instructions
def dynamic_instruction(ctx: RunContext[KnowledgeDeps]) -> str:
    print("dynamic instructions called")
    return ""

@kb_agent.tool
def get_knowledge_base(ctx: RunContext[KnowledgeDeps], query: str) -> str:
    """
    Retrieve relevant information from the knowledge base based on a search query.

    Args:
        query: The search query, usually a key or topic name or question.

    Returns:
        The matching information or a "not found" message.
    """
    print(query)
    return ctx.deps.kb

async def main():
    path = Path(__file__).parent / "data/small/kb.txt"
    deps = KnowledgeDeps(kb=path.read_text(encoding="utf-8"))

    message_history = []
    
    print("KB Agent is ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            break

        try:
            result = await kb_agent.run(
                user_input, 
                deps=deps,
                message_history=message_history
            )
        except Exception as e:
            print(f"Error: {e}")
            continue
        message_history = result.all_messages()        
        print(f"Agent: {result.output}")
        print(result.usage())

if __name__ == "__main__":
    asyncio.run(main())


# How long does international shipping take?
# which Leather jackets are available in the store?
# get the contact details of Harvest & Mill store
# What is the return policy?
# RunUsage(input_tokens=1745, output_tokens=221, requests=2, tool_calls=1)