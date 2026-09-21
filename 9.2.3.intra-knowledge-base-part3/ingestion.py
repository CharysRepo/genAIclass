import asyncio
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import asyncpg
from pgvector.asyncpg import register_vector
from pydantic_ai import Embedder
from dotenv import load_dotenv

load_dotenv(override=True)
embedder = Embedder('ollama:snowflake-arctic-embed2')

def chunk_text(text: str, file_name: str, chunk_size: int = 1024) -> list[dict[str, str]]:
        raw_doc = Document(page_content=text)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents([raw_doc])
        chunks = [doc.page_content for doc in split_docs]

        print(f"Created {len(chunks)} chunks: {file_name}")
        return chunks

async def setup_vector_database_and_ingest(pool):
    async with pool.acquire() as conn:
        print("Creating schema...")
        await conn.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                doc_name TEXT,
                chunk_id TEXT,
                chunk_content TEXT,
                embedding_vector vector(1024),
                search_vector tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', chunk_content)
                ) STORED
            );
                           
            CREATE INDEX IF NOT EXISTS idx_documents_embedding 
            ON document_chunks USING hnsw (embedding_vector vector_cosine_ops);
            
            CREATE INDEX IF NOT EXISTS idx_documents_search 
            ON document_chunks USING GIN(search_vector);
        """)
        
        dir = Path(__file__).parent / "data"
        print(f"Ingesting files from {dir}")

        for file in dir.glob("*"):
            text = file.read_text(encoding="utf-8")
            chunks = chunk_text(text, file.name)
            for i,chunk in enumerate(chunks):
                result = await embedder.embed_documents(chunk)
                chunk_id = f"{file.stem}-chunk-{i}"
                await conn.execute(
                    "INSERT INTO document_chunks (doc_name, chunk_id, chunk_content, embedding_vector) VALUES ($1, $2, $3, $4)", 
                    file.name, chunk_id, chunk, result.embeddings[0]
                )
        print("Setup complete.")

async def main():
    DSN = "postgresql://postgres:postgres@localhost:5432/postgres"
    pool = await asyncpg.create_pool(DSN, init=register_vector)
    try:
        await setup_vector_database_and_ingest(pool)
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
