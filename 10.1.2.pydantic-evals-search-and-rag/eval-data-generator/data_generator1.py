import asyncio
import asyncpg
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

class GoldenRecord(BaseModel):
    query: str = Field(description="A realistic search query.")
    expected_answer: str = Field(description="The ideal answer grounded in the chunk.")
    ground_truth_chunk_id: str = Field(description="The chunk_id from the source DB.")

agent = Agent(
    "ollama:glm-4.7-flash:q4_K_M",
    output_type=GoldenRecord,
    instructions="You are a search quality engineer. Create a query/answer pair for the provided text.",
)

CONCURRENT_REQUESTS = 5
DSN = "postgresql://postgres:postgres@localhost:5432/postgres"


async def setup_db(pool):
    async with pool.acquire() as conn:
        print("🛠️ Ensuring golden_records table exists...")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS golden_records1 (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                expected_answer TEXT NOT NULL,
                ground_truth_chunk_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Optional: Add an index for faster lookups during generation checks
            CREATE INDEX IF NOT EXISTS idx_gt_chunk_id ON golden_records1(ground_truth_chunk_id);
        """
        )


async def generate_and_store(row, pool, semaphore):
    async with semaphore:
        chunk_id = row["chunk_id"]
        content = row["chunk_content"]

        try:
            result = await agent.run(f"Chunk ID: {chunk_id}\nContent: {content}")
            data = result.output

            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO golden_records1 (query, expected_answer, ground_truth_chunk_id)
                    VALUES ($1, $2, $3)
                    """,
                    data.query,
                    data.expected_answer,
                    data.ground_truth_chunk_id,
                )
            return True
        except Exception as e:
            print(f"❌ Error for chunk {chunk_id}: {e}")
            return False


async def main():
    pool = await asyncpg.create_pool(dsn=DSN)
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    try:
        await setup_db(pool)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT chunk_id, chunk_content 
                FROM document_chunks 
                WHERE chunk_id NOT IN (SELECT ground_truth_chunk_id FROM golden_records1)
            """
            )

        if not rows:
            print("🙌 No new chunks to process. Everything is golden!")
            return

        print(f"🚀 Generating golden data for {len(rows)} chunks...")
        tasks = [generate_and_store(row, pool, semaphore) for row in rows]
        results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r)
        print(f"✨ Finished! Saved {success_count} records to 'golden_records1'.")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
