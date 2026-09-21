from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings
from config.config_reader import settings
import os

model = OpenAIChatModel(settings.OCR_MODEL, provider=OllamaProvider(api_key=os.getenv("OLLAMA_API_KEY"), base_url=os.getenv("OLLAMA_BASE_URL")), settings=ModelSettings(temperature=0))
image_to_text_agent = Agent(
    model=model,
    instructions="""
    You are an OCR expert specialized in the text  extraction.
    You are always precise and make sure that extraction is done properly.
    """
)