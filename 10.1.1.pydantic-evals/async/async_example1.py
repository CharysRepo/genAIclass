import asyncio
from pydantic_ai import Agent
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    "ollama:glm-4.7-flash:q4_K_M", 
    instructions='Summarize the input.'
)

async def task(inp: str) -> str:
    res = await agent.run(inp)
    return res.output


dataset = Dataset(
    name='summarization_test',
    cases=[
        Case(
            name='Case1',
            inputs='PydanticAI is a Python Agent Framework.', 
            expected_output='PydanticAI summarizes as an agent framework for python.'
        ),
    ],
    evaluators=[EqualsExpected()],
)

async def main():
    results = await dataset.evaluate(task)    
    results.print(include_expected_output=True, include_output=True, include_input=True)

if __name__ == '__main__':
    asyncio.run(main())
