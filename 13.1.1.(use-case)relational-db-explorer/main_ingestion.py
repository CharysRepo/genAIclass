from services.ingestion_service import IngestionService
import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)

async def main():
    async with IngestionService() as service:
        await service.run_pipeline()    
        print("🚀 Data Ingestion complete.")

if __name__ == "__main__":
    asyncio.run(main())