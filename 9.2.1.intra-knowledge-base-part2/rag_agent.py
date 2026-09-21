import asyncio
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire
from dataclasses import dataclass
from hybrid_search_service import HybridSearchService

logfire.configure()
logfire.instrument_pydantic_ai()


@dataclass
class KnowledgeDeps:
    hybrid_search_service: HybridSearchService

kb_agent = Agent(
    #'groq:llama-3.3-70b-versatile',
     "ollama:glm-4.7-flash:q4_K_M",
    #instructions="You are a helpful assistant. Always use the tool ask_knowledge_base to get information before answering."
    instructions="You are a helpful assistant. Answer questions based on knowledge context provided to you"
)

@kb_agent.instructions
async def dynamic_instructions(ctx: RunContext[KnowledgeDeps]) -> str:
    print(ctx.prompt)
    results = await ctx.deps.hybrid_search_service.search(ctx.prompt)
    final_text = "\n".join([result["doc_content"] for result in results])
    return final_text

# @kb_agent.tool
async def ask_knowledge_base(ctx: RunContext[KnowledgeDeps], query: str) -> str:
    """
    Retrieve relevant information from the knowledge base based on a search query.

    Args:
        query: The search query, usually a key or topic name or question.

    Returns:
        The matching information or a "not found" message.
    """
    print(query)
    results = await ctx.deps.hybrid_search_service.search(query)
    final_text = "\n".join([result["doc_content"] for result in results])
    return final_text

async def main():
    deps = KnowledgeDeps(hybrid_search_service=HybridSearchService())
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