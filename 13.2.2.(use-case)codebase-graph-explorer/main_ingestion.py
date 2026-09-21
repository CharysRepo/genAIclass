from services.ingestion_service import AsyncIngestionService
from services.neo4j_db_service import AsyncNeo4jDBService
from dotenv import load_dotenv
load_dotenv(override=True)
from config.config_reader import settings
import asyncio

async def main():
    db_service = AsyncNeo4jDBService()
    await db_service.setup_database()
    ingestion_service = AsyncIngestionService(db_service)
    await ingestion_service.ingest_codebase(
            dir=settings.REPO_PATH
        )        
    print(f"Successfully processed the codebase.")
    await db_service.close()

if __name__ == "__main__":
    asyncio.run(main())
        