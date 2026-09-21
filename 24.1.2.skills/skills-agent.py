from dotenv import load_dotenv
load_dotenv(override=True)
from pathlib import Path
import logfire
import uvicorn
from pydantic_ai import Agent
from pydantic_ai_harness.skills import Skills

logfire.configure()
logfire.instrument_pydantic_ai()

skills_dir = Path(__file__).parent / 'skills'

agent = Agent(
    model="ollama:qwen3.5:9b",
    instructions='You are a helpful research assistant.',
    capabilities=[Skills(skills_dir)],
)

app = agent.to_web()

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=7932)

# write a short introduction for a blog about Python async programming
# OK, then use your content research skill to help me research async Python