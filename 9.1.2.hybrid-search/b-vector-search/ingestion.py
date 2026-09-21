import asyncio
from pathlib import Path
import asyncpg
from pgvector.asyncpg import register_vector
from pydantic_ai import Embedder
from dotenv import load_dotenv

load_dotenv(override=True)
embedder = Embedder('ollama:snowflake-arctic-embed2')

async def setup_vector_database_and_ingest(pool):
    async with pool.acquire() as conn:
        print("Creating schema...")
        await conn.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS documents2 (
                id SERIAL PRIMARY KEY,
                doc_name TEXT,
                doc_content TEXT,
                embedding_vector vector(1024)
            );
            CREATE INDEX IF NOT EXISTS idx_documents2_embedding 
            ON documents2 USING hnsw (embedding_vector vector_cosine_ops);
        """)
        
        dir = Path(__file__).parent.parent / "data"
        print(f"Ingesting files from {dir}")

        for file in dir.glob("*"):
            text = file.read_text(encoding="utf-8")
            result = await embedder.embed_documents(text)
            await conn.execute(
                "INSERT INTO documents2 (doc_name, doc_content, embedding_vector) VALUES ($1, $2, $3)", 
                file.name, text, result.embeddings[0]
            )
        print("Vector database setup complete.")

async def main():
    DSN = "postgresql://postgres:postgres@localhost:5432/postgres"
    pool = await asyncpg.create_pool(DSN, init=register_vector)
    try:
        await setup_vector_database_and_ingest(pool)
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
