import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from services.pdf_conversion_service import PdfToTextConversionService
from db.database_manager import DatabaseManager
from config.config_reader import settings
from agents.embedder import embedder

class IngestionService:
    def __init__(self):
        self.pdf_conversion_service = PdfToTextConversionService()
        self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS_EMBEDDING)

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
                "file_name": file_name,
                "chunk_content": doc.page_content
            })

        print(f"Extracted and created {len(chunks)} chunks: {file_name}")
        return chunks

    async def store_chunks(self, chunks: list[dict[str, str]]) -> None:
        async with self.semaphore:
            result = await embedder.embed_documents([f"{c["chunk_content"]}" for c in chunks])
        
        pool = DatabaseManager.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for i, chunk in enumerate(chunks, start=1):
                    file_name = chunk["file_name"]
                    file_stem = file_name.rsplit(".", 1)[0]
                    await conn.execute(
                        "INSERT INTO doc_chunks (doc_id, chunk_id, chunk_content, embedding_vector) VALUES ($1, $2, $3, $4)",
                        file_name, f"{file_stem}-chunk-{i}", chunk["chunk_content"], result.embeddings[i-1]
                    )
        print(f"✅ Stored {len(chunks)} chunks.")

    async def extract_and_chunk(self, file:Path) -> str:
        chunks = None
        if file.suffix.lower() == ".pdf":
            text = await self.pdf_conversion_service.convert_pdf_to_text(file)
            chunks = await self.chunk_text(text, file_name=file.name, separators=["\n\n--- Page Break ---\n\n", "\n## ", "\n### ", "\n\n", "\n", " "])
        elif file.suffix.lower() == ".md":
            text = file.read_text(encoding="utf-8")
            chunks = await self.chunk_text(text, file_name=file.name, separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""])
        return chunks      
    
    async def ingest_file(self, file:Path) -> str:
        chunks = await self.extract_and_chunk(file)
        if chunks:
            await self.store_chunks(chunks)
        return f"Ingested {file.name} with {len(chunks) if chunks else 0} chunks."

    async def ingest_directory(self, dir:Path) -> list[str]:
        tasks = []
        for file in dir.rglob("*"):
            if file.is_file():
                tasks.append(self.extract_and_chunk(file))
        chunks_list = await asyncio.gather(*tasks)

        tasks = []
        for chunks in chunks_list:
            if chunks:
                tasks.append(self.store_chunks(chunks))
        await asyncio.gather(*tasks)

    async def remove_file(self, file:Path):
        pool = DatabaseManager.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    "DELETE FROM doc_chunks WHERE doc_id = $1", 
                    file.name
                )
                print(f"Records cleared for {file.name}: {result}")

