from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire
import asyncio
from db.database_manager import DatabaseManager
from services.ingestion_service import IngestionService
from services.pdf_conversion_service import PdfToTextConversionService

logfire.configure()
logfire.instrument_pydantic_ai()

async def main():
    await DatabaseManager.initialize()
    ingestion_service = IngestionService()

    docs_dir = Path(__file__).parent / "data/samples"
    await ingestion_service.ingest_directory(docs_dir)

    await DatabaseManager.disconnect()

if __name__ == "__main__":
    asyncio.run(main())