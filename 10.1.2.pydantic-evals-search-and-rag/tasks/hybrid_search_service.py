import asyncpg
from pgvector.asyncpg import register_vector
from pydantic_ai import Embedder

class HybridSearchService:
    def __init__(self):
        self.embedder = Embedder('ollama:snowflake-arctic-embed2')

    async def search(self, user_input, limit=5, k=60): 
        DSN = "postgresql://postgres:postgres@localhost:5432/postgres"
        pool = await asyncpg.create_pool(DSN, init=register_vector)   
        async with pool.acquire() as conn:
            result = await self.embedder.embed_query(user_input)
            query_vector = result.embeddings[0]

            vector_search_query = """
                SELECT chunk_id, chunk_content, (embedding_vector <=> $1) AS score
                FROM document_chunks
                ORDER BY score
                LIMIT $2; 
            """
            vector_rows = await conn.fetch(vector_search_query, query_vector, limit)

            kw_search_query = """
                SELECT chunk_id, chunk_content, ts_rank(search_vector, query) as score
                FROM document_chunks, plainto_tsquery('english', $1) query
                WHERE search_vector @@ query
                ORDER BY score DESC
                LIMIT $2;  
            """
            kw_rows = await conn.fetch(kw_search_query, user_input, limit)

            chunk_lookup = {}
            scores = {}

            for rank, row in enumerate(vector_rows, start=1):
                name = row['chunk_id']
                chunk_lookup[name] = row['chunk_content'] 
                scores[name] = scores.get(name, 0) + (1 / (k + rank))

            for rank, row in enumerate(kw_rows, start=1):
                name = row['chunk_id']
                chunk_lookup[name] = row['chunk_content'] 
                scores[name] = scores.get(name, 0) + (1 / (k + rank))

            final_results = []
            sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

            for chunk_id, score in sorted_keys:
                content = chunk_lookup[chunk_id]
                print(f"chunk_id={chunk_id} | Score={score:.4f}")
                final_results.append({
                    "score": score,
                    "chunk_id": chunk_id,
                    "chunk_content": content
                })

            return final_results
