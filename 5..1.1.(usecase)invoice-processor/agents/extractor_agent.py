from pathlib import Path
from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.github import GitHubProvider
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.settings import ModelSettings
from tqdm.asyncio import tqdm_asyncio
import asyncio
import os
import logfire
from dotenv import load_dotenv
from models.invoice import Invoice
from utils.utils import pdf_to_jpg, list_all_files

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

model = OpenAIChatModel("openai/gpt-4o", provider=GitHubProvider(api_key=os.getenv("GITHUB_API_KEY")), settings=ModelSettings(temperature=0))
#model = GroqModel('meta-llama/llama-4-scout-17b-16e-instruct', provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY")), settings=ModelSettings(temperature=0))

extraction_agent = Agent(
    model=model,
    output_type=Invoice,  
)

@extraction_agent.instructions
def dynamic_instruction(ctx:RunContext[str]) -> str:
     return f"""You are an AI assistant specialized in extracting information from invoice images for the company {ctx.deps}.
        Your task is to analyze the given invoice image and extract the relevant information.
        If a field is not present in the invoice or cannot be determined leave it as null.
        If the invoice date format is ambiguous, assume it's the international date format dd-mm-yyyy."""

async def run_inference(image_path: Path) -> Invoice:
        extractor_message_parts = ["Extract the structure from the provided document.", BinaryContent(data=image_path.read_bytes(), media_type="image/jpeg")]

        try:
            print(f"##File analyzed: {image_path.stem}")
            extraction_response = await extraction_agent.run(extractor_message_parts, deps="TechNova Solutions, Inc.")
            print(f"\n##Extracted data: \n{extraction_response.output}")
        except Exception as e:
            print(f"Unexpected error while inference: {e}")
            raise e
        return extraction_response.output

async def extract_invoices_data(invoices_dir:Path, temp_dir:Path):
    invoices_filenames = []
    tasks = []
    print('Extracting the invoices data')
    pdf_paths = list_all_files(invoices_dir)
    for pdf_path in pdf_paths:
        print(pdf_path)
        invoices_filenames.append(pdf_path)
        image_paths = pdf_to_jpg(pdf_path, temp_dir)
        task = asyncio.create_task(run_inference(image_paths[0]))
        tasks.append(task)
    invoices_result = await tqdm_asyncio.gather(*tasks)
    return invoices_filenames, invoices_result 