# pip install ddgs
import asyncio
from dataclasses import dataclass
import os
from pydantic import BaseModel
from ddgs import DDGS
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

@dataclass
class SearchDeps:
    max_results: int = 5

search_agent = Agent(
    #'groq:llama-3.3-70b-versatile',
    #'github:openai/gpt-4o',
    "ollama:llama3-groq-tool-use:8b",
   #"ollama:qwen3.5:9b",
    deps_type=SearchDeps,
    instructions="You are a helpful assistant. Use the tools provided to find requested infomration."
)

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

@search_agent.tool
def web_search(ctx: RunContext[SearchDeps], query: str) -> list[SearchResult]:
    """
    Perform a web search.
    
    Args:
        query: The search query.
        max_results: The number of results to return (default 5).
    """
    results = []
    with DDGS() as ddgs:
        search_results = ddgs.text(query, max_results=ctx.deps.max_results)
        print(search_results)
        for r in search_results:
            results.append(SearchResult(
                title=r.get('title', ''),
                url=r.get('href', ''),
                snippet=r.get('body', '')
            ))
    print(results)
    return results

class Deps:
    pass
async def main(user_prompt):
    search_deps = Deps()
    response = await search_agent.run(user_prompt, deps=search_deps)
    print(response.output)

if __name__ == "__main__":
    user_prompt = "search web for the top 3 news stories in GenerativeAI today?"
    asyncio.run(main(user_prompt))

# "search web for the top 3 news stories in GenerativeAI today?"
# "search web for the latest updates on israel, usa and iran conflict"