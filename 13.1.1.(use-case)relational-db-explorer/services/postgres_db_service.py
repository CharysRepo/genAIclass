import asyncio
import re
import time
import asyncpg
from custom_types.types import (
    ColumnInfo,
    ForeignKeyInfo,
    QueryResult,
    SchemaInfo,
    TableInfo
)
from config.config_reader import settings

class PostgresDBService():
    FORBIDDEN_KEYS = {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE",
        "VACUUM",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "COPY",
        "MERGE",
        "CALL",
        "DO",
    }

    def __init__(self, host: str=settings.POSTGRES_HOST, user: str=settings.POSTGRES_USERNAME, password: str=settings.POSTGRES_PASSWORD, port:int = settings.POSTGRES_PORT, database: str=settings.POSTGRES_DATABASE, read_only: bool = True) -> None:
        self.pool: asyncpg.Pool | None = None
        self.read_only = read_only
        self.dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    async def __aenter__(self) -> "PostgresDBService":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(
        self,
        min_size: int = 1,
        max_size: int = 10,
        command_timeout: float = 60.0,
        timeout: float = 120.0,
    ) -> asyncpg.Pool:
        """
        Connect to the database.

        Args:
            min_size: Minimum size of the connection pool.
            max_size: Maximum size of the connection pool.
            command_timeout: Timeout for individual queries in seconds.
            timeout: Timeout for establishing the connection in seconds.
        """
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=min_size,
                max_size=max_size,
                command_timeout=command_timeout,  # Timeout for individual queries
                timeout=timeout,  # Timeout for establishing connection
            )
        return self.pool

    async def close(self) -> None:
        """Close database connection."""
        if self.pool:
            await self._pool.close()
            self.pool = None

    def _prepare_query(self, query: str) -> str:
        """Strips comments and normalizes whitespace."""
        # Strip block comments
        sql = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
        # Strip line comments
        sql = re.sub(r"--.*", " ", sql)
        # Normalize whitespace
        return " ".join(sql.split()).strip()
    
    def _check_singular_query(self, cleaned_query: str) -> None:
        """Ensures only one SQL statement is present."""
        if ";" in cleaned_query.rstrip(";"):
            raise PermissionError("Multiple statements are not allowed for security reasons.")
        
    def _check_permissions(self, cleaned_query: str) -> None:
        if self.read_only:
            upper_sql = cleaned_query.upper()
            if any(upper_sql.startswith(kw) for kw in self.FORBIDDEN_KEYS):
                    raise ValueError("Write operation denied. Database is in read-only mode.")

    def check_query_safety(self, query: str) -> str:
        """Check if the query contains forbidden keywords."""
        prepared_query = self._prepare_query(query)

        if not prepared_query:
            raise ValueError("Query is empty or only contains comments.")

        self._check_singular_query(prepared_query)
        self._check_permissions(prepared_query)

        return prepared_query

    async def execute(self, query: str) -> str | QueryResult:
        """Execute a SQL query with optional parameters."""
        try:
            safe_query = self.check_query_safety(query)
            if safe_query is None:
                raise ValueError("Query is not allowed in read-only mode")

            start_time = time.perf_counter()
            records = await self.pool.fetch(safe_query)

            # Transform data to fit schema
            processed_rows = [tuple(row) for row in records]
            columns = list(records[0].keys()) if records else []

            return QueryResult(
                columns=columns,
                rows=processed_rows,
                row_count=len(processed_rows),
                execution_time_ms=(time.perf_counter() - start_time),
            )
        except Exception as e:
            return f"SQL Execution Error: {str(e)}"
    

    async def get_tables(self) -> list[str]:
        """Get list of tables in the public schema."""
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """
        result = await self.pool.fetch(query)
        table_list = [r["table_name"] for r in result]

        return table_list

    async def get_foreign_keys(self, table_name: str) -> list[ForeignKeyInfo]:
        """Get information about foreign keys in given table"""
        # Use $1 instead of f-string
        query = """
        SELECT DISTINCT
            kcu.column_name AS local_column,
            ccu.table_name AS foreign_table,
            ccu.column_name AS foreign_column
        FROM
            information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_name = $1;
        """

        # Pass table_name as an argument to fetch
        records = await self.pool.fetch(query, table_name)

        return [
            ForeignKeyInfo(
                column=r["local_column"],
                references_table=r["foreign_table"],
                references_column=r["foreign_column"],
            )
            for r in records
        ]

    async def get_table_info(self, table_name: str) -> TableInfo | None:
        """Get detailed information about a specific table."""
        tables = await self.get_tables()
        if table_name not in tables:
            return None

        query = """
            SELECT DISTINCT ON (c.ordinal_position)
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                CASE WHEN tc.constraint_type = 'PRIMARY KEY' THEN TRUE ELSE FALSE END AS is_primary
            FROM information_schema.columns c
            LEFT JOIN information_schema.key_column_usage kcu
                ON c.table_name = kcu.table_name
                AND c.column_name = kcu.column_name
            LEFT JOIN information_schema.table_constraints tc
                ON kcu.constraint_name = tc.constraint_name
                AND tc.constraint_type = 'PRIMARY KEY'
            WHERE c.table_name = $1
            AND c.table_schema = 'public'
            ORDER BY c.ordinal_position, is_primary DESC;
        """
        records = await self.pool.fetch(query, table_name)

        columns = []
        primary_keys = []
        foreign_keys = []

        for r in records:
            col = ColumnInfo(
                name=r["column_name"],
                data_type=r["data_type"],
                nullable=(r["is_nullable"] == "YES"),
                default=r["column_default"],
                is_primary_key=r["is_primary"],
            )

            if col.is_primary_key:
                primary_keys.append(col.name)
            columns.append(col)

        # Get foreign keys using your existing method
        foreign_keys = await self.get_foreign_keys(table_name)

        table = TableInfo(
            name=table_name,
            columns=columns,
            foreign_keys=foreign_keys,
            primary_key=primary_keys,
        )
        return table

    async def get_schema(self) -> SchemaInfo:
        """Get database schema information."""
        table_names = await self.get_tables()

        tasks = [self.get_table_info(table_name) for table_name in table_names]
        tables = await asyncio.gather(*tasks)

        return SchemaInfo(tables=[t for t in tables if t])

    async def explain(self, query: str) -> str:
        """Get query execution plan."""
        query = f"EXPLAIN {query}"

        explanation = ""
        try:
            result = await self.pool.fetch(query)
            for r in result:
                explanation += f"{r['QUERY PLAN']}"
            return explanation

        except asyncpg.exceptions.PostgresSyntaxError:
            return "Invalid query, please try again"