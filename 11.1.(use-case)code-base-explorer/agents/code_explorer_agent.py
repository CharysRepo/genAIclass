from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from services.hybrid_search_service import HybridSearchService
from config.config_reader import settings

@dataclass
class KnowledgeDeps:
    hybrid_search_service: HybridSearchService

code_explorer_agent = Agent(
    settings.CHAT_MODEL,
    instructions="""
    You are a helpful assistant. Answer questions based on the information provided in the context below.
    """
)

@code_explorer_agent.instructions
async def dynamic_context(ctx: RunContext[KnowledgeDeps]) -> str:
    print(ctx.prompt)
    search_results = await ctx.deps.hybrid_search_service.search(ctx.prompt, limit=3)
    print(search_results)

    formatted = []
    for result in search_results:
        formatted.append(
            f"[source:{result.file_id}]\n\ncontent:{result.chunk_content}")
    final_text = "\n\n---\n\n".join(formatted)
    return f"<context>{final_text}</context>"

async def retrieve_documents(ctx: RunContext[KnowledgeDeps], query: str, top_k: int = 20) -> str:
    """
    Search knowledge base for relevant documents.

    Args:
        query: The search query — rephrase the user's question as a search query.
        top_k: Number of documents to retrieve.
    """
    print(query)
    search_results = await ctx.deps.hybrid_search_service.search(query, limit=top_k)
    print(search_results)

    formatted = []
    for result in search_results:
        formatted.append(
            f"[source:{result.file_id}]\n\ncontent:{result.chunk_content}"
        )
    return "\n\n---\n\n".join(formatted)
