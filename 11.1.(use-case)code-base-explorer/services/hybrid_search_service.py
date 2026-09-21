import asyncio
from dataclasses import dataclass
from db.database_manager import DatabaseManager
from agents.embedder import embedder

@dataclass
class SearchResult:
    file_id: str
    chunk_id: str
    chunk_content: str
    chunk_summary: str
    rr: float = 0.0

class HybridSearchService:
    def __init__(self):
        pass
    
    def process_rows_rrf(self, rows_lists: list[list], limit:int, k: int) -> list[SearchResult]:
        merged_results = {}
        for rows in rows_lists:
            for rank, row in enumerate(rows, start=1):
                chunk_id = row['chunk_id']
                rr_score = 1.0 / (k + rank)
                
                if chunk_id not in merged_results:
                    merged_results[chunk_id] = SearchResult(
                        file_id = row["file_id"],
                        chunk_id=chunk_id, 
                        chunk_summary=row.get("chunk_summary", ""),
                        chunk_content=row.get("chunk_content", ""),
                        rr=rr_score
                    )
                else:
                    merged_results[chunk_id].rr += rr_score
        sorted_results = sorted(merged_results.values(), key=lambda x: x.rr, reverse=True)
        return sorted_results[:limit]

    async def search(self, query: str, limit: int = 3, k: int = 60) -> list[SearchResult]:
        pool = await DatabaseManager.get_pool()
        kw_search_query = """
                SELECT file_id, chunk_id, chunk_content, chunk_summary, ts_rank(search_vector, query) as score
                FROM code_chunks, plainto_tsquery('english', $1) query
                WHERE search_vector @@ query
                ORDER BY score DESC
                LIMIT $2;  
            """
        vector_search_query = """
                SELECT file_id, chunk_id, chunk_content, chunk_summary, (embedding_vector <=> $1) AS score
                FROM code_chunks
                ORDER BY score
                LIMIT $2; 
            """
                
        tasks = []
        tasks.append(pool.fetch(kw_search_query, query, limit))
        res = await embedder.embed_query(query)
        query_vector = res.embeddings[0]
        tasks.append(pool.fetch(vector_search_query, query_vector, limit))

        rows_lists  = await asyncio.gather(*tasks)
        # print(rows_lists)
        return self.process_rows_rrf(rows_lists, limit, k)