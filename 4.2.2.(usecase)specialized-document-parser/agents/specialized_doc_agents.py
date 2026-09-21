import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent
from models.invoice import Invoice
from models.payslip import Payslip
from models.drivers_license import DriversLicense
from models.category import DocumentCategory
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.github import GitHubProvider
from pydantic_ai.settings import ModelSettings

load_dotenv(override=True)

model = OpenAIChatModel("openai/gpt-4.1", provider=GitHubProvider(api_key=os.getenv("GITHUB_API_KEY")), settings=ModelSettings(temperature=0))
model = GroqModel('meta-llama/llama-4-scout-17b-16e-instruct', provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY")), settings=ModelSettings(temperature=0))

classifier_agent = Agent(
    model,
    instructions="You are a helpful assistant.",
    output_type=DocumentCategory
)

extraction_agent = Agent(
    model,
    instructions="You are a helpful assistant."
)

# Map classification types to schemas
classification_to_schema = {
    "invoice": Invoice,
    "payslip": Payslip,
    "driver_license": DriversLicense,
}

async def run_inference(image_path: Path) -> Invoice | Payslip | DriversLicense:
    classifier_message_parts = ["Classify the type of the following document based on its content and structure.", BinaryContent(data=image_path.read_bytes(), media_type="image/jpeg")]
    extractor_message_parts = ["Extract the structure from the provided document.", BinaryContent(data=image_path.read_bytes(), media_type="image/jpeg")]

    try:
        print(f"\n##File analyzed: {image_path.stem}")
        classifier_response = await classifier_agent.run(classifier_message_parts)
        category = classifier_response.output.category
        print(f"\n##Detected category: {category}")
        output_type = classification_to_schema.get(category)
        extraction_response = await extraction_agent.run(extractor_message_parts, output_type=output_type)
        print(f"\n##Extracted data: \n{extraction_response.output}")
    except Exception as e:
        print(f"Unexpected error while inference: {e}")
        raise e
    return extraction_response.output

