from tqdm.asyncio import tqdm_asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.github import GitHubProvider
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.settings import ModelSettings
from models.generic import ModelAnalysisOutput, OCROutput

load_dotenv()

model = OpenAIChatModel("openai/gpt-4o", provider=GitHubProvider(api_key=os.getenv("GITHUB_API_KEY")), settings=ModelSettings(temperature=0))
model = GroqModel('meta-llama/llama-4-scout-17b-16e-instruct', provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY")), settings=ModelSettings(temperature=0))

agent = Agent(
    model=model,
    instructions="You are an OCR expert specialized in the data extraction \
        from various types of documents. You are always precise and make sure \
        that extraction is done properly.",
    output_type=ModelAnalysisOutput,  
)

async def run_inference(image_path: Path) -> OCROutput:
        message_parts = ["Extract the structure from the provided document.", BinaryContent(data=image_path.read_bytes(), media_type="image/jpeg")]

        try:
            analysis_result = await agent.run(message_parts)
            print(f"File analyzed: {image_path.stem}")
        except Exception as e:
            print(f"Unexpected error while inference: {e}")
            raise e
        
        output = OCROutput(filename=image_path.stem, analysis_result=analysis_result.output)
        return output


async def analyze_each_page(image_paths: list[Path]) -> list[OCROutput]:
    tasks = [run_inference(path) for path in image_paths]
    results_objects = await tqdm_asyncio.gather(*tasks)
    return results_objects