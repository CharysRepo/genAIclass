import asyncpg
from pgvector.asyncpg import register_vector
from config.config_reader import settings

class DatabaseManager:
    _pool = None

    @classmethod
    def get_pool(cls):
        if not cls._pool:
            raise RuntimeError("DatabaseManager not initialized. Call await DatabaseManager.initialize() first.")
        return cls._pool

    @classmethod
    async def initialize(cls, vector_dim: int = settings.VECTROR_DIM):
        if cls._pool: return cls._pool        
        dsn = str(settings.DATABASE_URL)        
        cls._pool = await asyncpg.create_pool(dsn, init=register_vector)
        await cls._setup_schema(vector_dim)
    
    @classmethod
    async def _setup_schema(cls, vector_dim: int):
        async with cls._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS code_chunks (
                        id SERIAL PRIMARY KEY,
                        file_id TEXT,
                        chunk_id TEXT,
                        chunk_content TEXT,
                        chunk_summary TEXT,
                        embedding_vector vector({vector_dim}),
                        search_vector tsvector
                            GENERATED ALWAYS AS (                                
                                to_tsvector('english', coalesce(chunk_summary, '') || ' ' || coalesce(chunk_content, ''))
                            ) STORED
                    );
                """)

                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_code_chunks_chunk_id 
                    ON code_chunks (chunk_id);
                """)
                
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON code_chunks 
                    USING hnsw (embedding_vector vector_cosine_ops)
                    WITH (m = 16, ef_construction = 128);
                """)
                
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_chunks_search ON code_chunks USING gin (search_vector);                                   
                """)
        print("✅ Database schema initialized.")

    @classmethod
    async def disconnect(cls):
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
