import asyncio
from pathlib import Path
import asyncpg

async def setup_database_and_ingest(pool):
    async with pool.acquire() as conn:
        print("Creating schema...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents1 (
                id SERIAL PRIMARY KEY,
                doc_name TEXT,
                doc_content TEXT,
                search_vector tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', doc_content)
                ) STORED
            );
            
            CREATE INDEX IF NOT EXISTS idx_documents_search 
            ON documents1 USING GIN(search_vector);
        """)

        dir = Path(__file__).parent.parent / "data"
        print(f"Ingesting files from {dir}")

        for file in dir.glob("*"):
            text = file.read_text(encoding="utf-8")
            await conn.execute(
                "INSERT INTO documents1 (doc_name, doc_content) VALUES ($1, $2)", 
                file.name, text
            )
        
        print("Setup complete.")

async def main():
    DSN = "postgresql://postgres:postgres@localhost:5432/postgres"
    pool = await asyncpg.create_pool(DSN)    
    try:
        await setup_database_and_ingest(pool)
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
