import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from db.database_manager import DatabaseManager
from config.config_reader import settings
from agents.embedder import embedder

class IngestionService:
    def __init__(self):
        pass

    async def chunk_text(self, text: str, file_name: str, separators:list[str], chunk_size: int = settings.CHUNK_SIZE) -> list[dict[str, str]]:
        raw_doc = Document(page_content=text)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),
            separators=separators
        )
        split_docs = text_splitter.split_documents([raw_doc])

        chunks = []
        for doc in split_docs:
            chunks.append({
                "chunk_content": doc.page_content
            })

        print(f"Created {len(chunks)} chunks: {file_name}")
        return chunks

    async def store_chunks(self, file:Path, chunks: list[dict[str, str]]):
        result = await embedder.embed_documents([f"{c["chunk_content"]}" for c in chunks])
        
        pool = DatabaseManager.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for i, chunk in enumerate(chunks, start=1):
                    await conn.execute(
                        "INSERT INTO doc_chunks (doc_id, chunk_id, chunk_content, embedding_vector) VALUES ($1, $2, $3, $4)",
                        file.name, f"{file.stem}-chunk-{i}", chunk["chunk_content"], result.embeddings[i-1]
                    )
        print(f"✅ Stored {len(chunks)} chunks.")
    
    async def ingest_file(self, file:Path) -> str:
        # print(f"ingest_file:{file}")
        chunks = None
        if file.suffix.lower() == ".pdf":
            text = await self.pdf_conversion_service.convert_pdf_to_text(file)
            chunks = await self.chunk_text(text, file_name=file.name, separators=["\n\n--- Page Break ---\n\n", "\n## ", "\n### ", "\n\n", "\n", " "])
        elif file.suffix.lower() == ".md":
            text = file.read_text(encoding="utf-8")
            chunks = await self.chunk_text(text, file_name=file.name, separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""])
        else:
            return f"Not supported filetype:{file.name}"
        
        if chunks:
            await self.store_chunks(file, chunks)
            return f"ingestion of {file.name} completed"
        else:
            return f"Skipped {file.name}: Unsupported file type or empty content."

    async def ingest_directory(self, dir:Path) -> list[str]:
        tasks = []
        for file in dir.glob("*"):
            tasks.append(self.ingest_file(file))
        results = await asyncio.gather(*tasks)
        return results

    async def remove_file(self, file:Path):
        pool = DatabaseManager.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    "DELETE FROM chunks WHERE source = $1", 
                    file.name
                )
                print(f"Records cleared for {file.name}: {result}")

