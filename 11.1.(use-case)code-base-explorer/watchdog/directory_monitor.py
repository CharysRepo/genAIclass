import asyncio
from pathlib import Path
from watchfiles import watch, Change
from services.ingestion_service import IngestionService
from db.database_manager import DatabaseManager
from config.config_reader import settings

WATCHED_DIR = settings.WATCHED_DIR
ALLOWED_EXTENSIONS = settings.ALLOWED_EXTENSIONS

async def start_monitoring():
    await DatabaseManager.initialize(1024)

    service = IngestionService()
    print(f"Monitoring {WATCHED_DIR} for {ALLOWED_EXTENSIONS}...")

    for changes in watch(WATCHED_DIR):
        for change_type, path in changes:
            if not path.lower().endswith(ALLOWED_EXTENSIONS):
                continue

            if change_type == Change.added:
                print(f"Added: {path}")
                await service.ingest_file(Path(path))
                
            elif change_type == Change.modified:
                print(f"Modified: {path}")
                await service.remove_file(Path(path))
                await service.ingest_file(Path(path))
                
            elif change_type == Change.deleted:
                print(f"Deleted: {path}")
                await service.remove_file(Path(path))
    await DatabaseManager.disconnect()

if __name__ == "__main__":
    from dotenv import load_dotenv
    import logfire

    load_dotenv(override=True)
    logfire.configure()
    logfire.instrument_pydantic_ai()
    try:
        asyncio.run(start_monitoring())
    except KeyboardInterrupt:
        print("Monitoring stopped.")
