import asyncio
import asyncpg
import json
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dotenv import load_dotenv
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

class GoldenRecord(BaseModel):
    query: str = Field(description="A complex query that connects information from multiple chunks.")
    expected_answer: str = Field(description="The ideal synthesized answer.")
    ground_truth_chunk_ids: List[str] = Field(description="List of chunk_ids strictly from the provided context.")

agent = Agent(
    "ollama:glm-4.7-flash:q4_K_M",
    output_type=GoldenRecord,
    instructions=(
        "You are a search quality engineer. Create a multi-hop query/answer pair "
        "using ONLY the provided chunks. You must return the exact 'ID' strings "
        "for the chunks you used in the 'ground_truth_chunk_ids' field."
    ),
)

CONCURRENT_REQUESTS = 5
DSN = "postgresql://postgres:postgres@localhost:5432/postgres"

async def setup_db(pool):
    async with pool.acquire() as conn:
        print("🛠️ Ensuring golden_records table exists...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS golden_records2 (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                expected_answer TEXT NOT NULL,
                ground_truth_chunk_ids TEXT[] NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_golden_chunk_ids ON golden_records2 USING GIN (ground_truth_chunk_ids);
        """)

async def generate_and_store(batch, pool, semaphore):
    async with semaphore:
        # Extract valid IDs from the batch for verification
        valid_ids = {r['chunk_id'] for r in batch}
        context_str = "\n".join([f"ID: {r['chunk_id']} | Content: {r['chunk_content']}" for r in batch])
        
        try:
            result = await agent.run(f"Context Chunks:\n{context_str}")
            data = result.output

            # --- VALIDATION STEP ---
            # Filter out any IDs the LLM might have hallucinated
            sanitized_ids = [gid for gid in data.ground_truth_chunk_ids if gid in valid_ids]
            
            if not sanitized_ids:
                print(f"⚠️ Warning: No valid chunk IDs found in generated record for cluster {list(valid_ids)[:2]}...")
                return False

            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO golden_records2 (query, expected_answer, ground_truth_chunk_ids)
                    VALUES ($1, $2, $3)
                    """,
                    data.query, data.expected_answer, sanitized_ids
                )
            return True
        except Exception as e:
            print(f"❌ Error during generation/store: {e}")
            return False

async def main():
    pool = await asyncpg.create_pool(dsn=DSN)
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    try:
        await setup_db(pool)

        async with pool.acquire() as conn:
            print("🔍 Fetching semantic clusters via pgvector...")
            # Query fetches seed chunks and their 2 nearest semantic neighbors
            rows = await conn.fetch("""
                WITH seed_chunks AS (
                    SELECT chunk_id, chunk_content, embedding_vector
                    FROM document_chunks
                    WHERE chunk_id NOT IN (
                        SELECT unnest(ground_truth_chunk_ids) FROM golden_records2
                    )
                    LIMIT 30
                )
                SELECT 
                    json_agg(json_build_object(
                        'chunk_id', n.chunk_id, 
                        'chunk_content', n.chunk_content
                    )) as cluster
                FROM seed_chunks s
                CROSS JOIN LATERAL (
                    SELECT chunk_id, chunk_content
                    FROM document_chunks
                    ORDER BY embedding_vector <=> s.embedding_vector
                    LIMIT 3
                ) n
                GROUP BY s.chunk_id;
            """)

        if not rows:
            print("🙌 All data is processed!")
            return

        tasks = []
        for row in rows:
            cluster = row['cluster']
            # Ensure cluster is a Python list (handles variations in asyncpg/json-agg output)
            if isinstance(cluster, str):
                cluster = json.loads(cluster)
            tasks.append(generate_and_store(cluster, pool, semaphore))
            
        print(f"🚀 Processing {len(tasks)} clusters...")
        await asyncio.gather(*tasks)
        print("✨ Generation complete!")

    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
