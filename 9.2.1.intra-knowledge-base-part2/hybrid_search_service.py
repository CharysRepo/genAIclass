import asyncpg
from pgvector.asyncpg import register_vector
from pydantic_ai import Embedder

class HybridSearchService:
    def __init__(self):
        self.embedder = Embedder('ollama:snowflake-arctic-embed2')

    async def search(self, user_input, limit=3, k=60): 
        DSN = "postgresql://postgres:postgres@localhost:5432/postgres"
        pool = await asyncpg.create_pool(DSN, init=register_vector)   
        async with pool.acquire() as conn:
            result = await self.embedder.embed_query(user_input)
            query_vector = result.embeddings[0]

            vector_search_query = """
                SELECT doc_name, doc_content, (embedding_vector <=> $1) AS score
                FROM documents3
                ORDER BY score
                LIMIT $2; 
            """
            vector_rows = await conn.fetch(vector_search_query, query_vector, limit)

            kw_search_query = """
                SELECT doc_name, doc_content, ts_rank(search_vector, query) as score
                FROM documents3, plainto_tsquery('english', $1) query
                WHERE search_vector @@ query
                ORDER BY score DESC
                LIMIT $2;  
            """
            kw_rows = await conn.fetch(kw_search_query, user_input, limit)

            doc_lookup = {}
            scores = {}

            for rank, row in enumerate(vector_rows, start=1):
                name = row['doc_name']
                doc_lookup[name] = row['doc_content'] 
                scores[name] = scores.get(name, 0) + (1 / (k + rank))

            for rank, row in enumerate(kw_rows, start=1):
                name = row['doc_name']
                doc_lookup[name] = row['doc_content'] 
                scores[name] = scores.get(name, 0) + (1 / (k + rank))

            final_results = []
            sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

            for doc_name, score in sorted_keys:
                content = doc_lookup[doc_name]
                print(f"doc_name={doc_name} | Score={score:.4f}")
                final_results.append({
                    "doc_name": doc_name,
                    "score": score,
                    "doc_content": content
                })

            return final_results
