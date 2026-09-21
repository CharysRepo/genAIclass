import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from pathlib import Path
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from agents.embedder import embedder
from db.database_manager import DatabaseManager
from config.config_reader import settings

class SummaryDeps(BaseModel):
    doc_content: str
    
class IngestionService:
    language_map = {
            "py": Language.PYTHON,
            "js": Language.JS,
            "ts": Language.TS,
            "java": Language.JAVA,
            "cpp": Language.CPP,
            "go": Language.GO,
            "php": Language.PHP,
            "rb": Language.RUBY,
            "rs": Language.RUST,
            "swift": Language.SWIFT,
            "cs": Language.CSHARP
    }  

    def __init__(self):
        self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)  

    async def chunk_code(self, text: str, file_name: str, lang: Language, chunk_size: int = settings.CHUNK_SIZE) -> list[Document]:
        print(f"Chunking {file_name} with language={lang}")
        if lang:
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.1)
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.1)
            )
        
        raw_doc = Document(page_content=text)
        return text_splitter.split_documents([raw_doc])

    async def store_raw_chunks(self, split_docs: list[Document], file_name: str) -> list[str]:
        print(f"Storing raw chunks for {file_name} into the database")
        chunk_ids = []
        file_stem = file_name.rsplit('.', 1)[0]
        
        pool = DatabaseManager.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for i, doc in enumerate(split_docs, start=1):
                    cid = f"{file_stem}_chunk_{i}"
                    await conn.execute(
                        """INSERT INTO code_chunks 
                           (file_id, chunk_id, chunk_content) 
                           VALUES ($1, $2, $3)""",
                        file_name, cid, doc.page_content
                    )
                    chunk_ids.append(cid)
        return chunk_ids

    async def generate_summaries(self, file_name: str, chunk_ids: list[str], full_file_content: str):
        print(f"Generating summaries for {len(chunk_ids)} chunks of file {file_name}")
        summary_agent = Agent(
            settings.CHAT_MODEL, 
            deps_type=SummaryDeps,
            instructions="Situating chunks within a document for better retrieval."
        )

        @summary_agent.instructions
        def dynamic_instructions(ctx: RunContext[SummaryDeps]):
            return f"Summarize this chunk's role in the doc:\n<doc>{ctx.deps.doc_content}</doc>"

        deps = SummaryDeps(doc_content=full_file_content)
        pool = DatabaseManager.get_pool()

        async with pool.acquire() as conn:
            for cid in chunk_ids:
                row = await conn.fetchrow("SELECT chunk_content FROM code_chunks WHERE chunk_id = $1", cid)
                
                async with self.semaphore:
                    result = await summary_agent.run(f"<chunk>{row['chunk_content']}</chunk>", deps=deps)
                    summary = result.output

                await conn.execute(
                    "UPDATE code_chunks SET chunk_summary = $1 WHERE chunk_id = $2",
                    summary, cid
                )

    async def update_embeddings(self, chunk_ids: list[str]):
        pool = DatabaseManager.get_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT chunk_id, chunk_content, chunk_summary FROM code_chunks WHERE chunk_id = ANY($1)", 
                chunk_ids
            )
            
            texts_to_embed = [f"{r['chunk_summary']}\n\n{r['chunk_content']}" for r in rows]
            emb_result = await embedder.embed_documents(texts_to_embed)
            
            async with conn.transaction():
                for i, row in enumerate(rows):
                    await conn.execute(
                        "UPDATE code_chunks SET embedding_vector = $1 WHERE chunk_id = $2",
                        emb_result.embeddings[i], row['chunk_id']
                    )

    async def ingest_chunks_and_summaries(self, file: Path) -> None:
        text = file.read_text(encoding="utf-8")
        ext = file.suffix.lstrip('.')
        lang = self.language_map.get(ext)
        
        split_docs = await self.chunk_code(text, file.name, lang)
        chunk_ids = await self.store_raw_chunks(split_docs, file.name)
        
        await self.generate_summaries(file.name, chunk_ids, text)        
        return chunk_ids
    
    async def ingest_file(self, file:Path) -> None:
        chunk_ids = await self.ingest_chunks_and_summaries(file)
        if chunk_ids:
            print(f"Summaries complete. Starting batch embedding for {len(chunk_ids)} chunks...")
            await self.update_embeddings(chunk_ids)
            print("✅ File ingestion and embedding complete.")
        

    async def ingest_directory(self, dir:Path) -> None:
        tasks = []
        for file in dir.rglob("*"):
            if file.is_file():
                tasks.append(self.ingest_file(file))
        results = await asyncio.gather(*tasks)
        
        all_chunk_ids = [cid for sublist in results if sublist for cid in sublist]
        
        if all_chunk_ids:
            print(f"Summaries complete. Starting batch embedding for {len(all_chunk_ids)} chunks...")
            await self.update_embeddings(all_chunk_ids)
            print("✅ Directory ingestion and embedding complete.")
        

    async def remove_file(self, file:Path):
        pool = DatabaseManager.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    "DELETE FROM code_chunks WHERE source = $1", 
                    file.name
                )
                print(f"Records cleared for {file.name}: {result}")
