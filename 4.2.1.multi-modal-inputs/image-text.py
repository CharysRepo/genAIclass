import asyncio
from pathlib import Path
from pydantic_ai import Agent, BinaryContent
import logfire
from dotenv import load_dotenv

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent('groq:meta-llama/llama-4-scout-17b-16e-instruct', instructions='You are a helpful assistant.')

async def main():
    image_path = Path(__file__).parent.resolve() / 'data' /'image1.png'

    result = await agent.run(
        [
            'Describe the content of the following image.',
            BinaryContent(data=image_path.read_bytes(), media_type='image/jpeg'),
        ]
    )
    print(result.output)

if __name__ == '__main__':
    asyncio.run(main())