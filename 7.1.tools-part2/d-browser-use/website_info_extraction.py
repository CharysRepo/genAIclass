# pip install playwright
# playwright install chromium
from bs4 import BeautifulSoup
from html_to_markdown import convert_to_markdown
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
from playwright.async_api import async_playwright
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

@dataclass
class BrowserDeps:
    headless: bool = False 
    
browser_agent = Agent(
    #'groq:llama-3.3-70b-versatile',
    #'github:openai/gpt-4o',
    "ollama:llama3-groq-tool-use:8b",
    #"ollama:qwen3.5:9b",
    deps_type=BrowserDeps,
    instructions="You are a helpful assistant. Use the tools provided to find requested infomration."
)

@browser_agent.tool
async def open_web_site(ctx: RunContext[BrowserDeps], url: str) -> str:
    """Navigates to a URL."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=ctx.deps.headless)
        page = await browser.new_page()
        await page.goto(url)
        html = await page.content()
        await asyncio.sleep(5)
        await browser.close()
    soup = BeautifulSoup(html, "html.parser")
    for noise in soup(["script", "style", "nav", "footer", "header", "aside"]):
        noise.decompose()        
    main_content = soup.find("main") or soup.body

    md = convert_to_markdown(str(main_content))
    print(md)
    return md
    
async def main(user_prompt): 
    deps = BrowserDeps(headless=False)
    result = await browser_agent.run(user_prompt, deps=deps)
    print(result.output)

if __name__ == "__main__":
    user_prompt =  'navigate http://ai.pydantic.dev and create a short summary of agent creation.'
    asyncio.run(main(user_prompt))

# 'navigate http://ai.pydantic.dev and create a short summary of agent creation.'
# 'navigate http://ai.pydantic.dev and summarize it in two sentences.'