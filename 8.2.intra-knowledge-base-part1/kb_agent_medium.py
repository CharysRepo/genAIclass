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
    instructions="You are a helpful assistant. Always use the tool ask_knowledge_base to get information before answering."
)


@kb_agent.tool
def ask_knowledge_base(ctx: RunContext[KnowledgeDeps], query: str) -> str:
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
    texts = []
    dir = Path(__file__).parent / "data/medium"
    for file in dir.glob("*"):
        texts.append(file.read_text(encoding="utf-8"))
    final_text = "\n".join(texts)
    print(f"length of final_text:{len(final_text)}")
    print(f"input tokens of final_text:{len(final_text)//4}")
    deps = KnowledgeDeps(kb=final_text)

    message_history = []
    
    print("RAG Agent is ready! (Type 'exit' to quit)")
    
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

if __name__ == "__main__":
    asyncio.run(main())

# Can I use ChatGPT for my assignment?
# What happens if I get sick during an exam?
# How is the AI & Machine Learning course graded?