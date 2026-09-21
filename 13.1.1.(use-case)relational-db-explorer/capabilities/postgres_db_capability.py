from typing import Any
from pydantic_ai import AgentToolset, FunctionToolset, ModelRetry,  RunContext
from pydantic_ai.capabilities import AbstractCapability
from dataclasses import dataclass
from pydantic_ai._instructions import AgentInstructions
from dataclasses import dataclass
from typing import Any, List
from pydantic_ai import RunContext
from services.postgres_db_service import PostgresDBService
from custom_types.types import (
    QueryResult,
    SchemaInfo,
    TableInfo
)

@dataclass
class DBDeps:
    db_service: PostgresDBService

@dataclass 
class PostgresCapability(AbstractCapability[DBDeps]): 

    def get_instructions(self) -> AgentInstructions[Any] | None:
        return """You are a helpful assistant for querying a Postgres database. 
        While converting natural language questions to valid Postgres SQL queries, follow these guidelines:
        1. Use only the tables and columns mentioned in the schema.
        2. Use proper JOIN clauses when querying multiple tables.
        3. Return ONLY the SQL query without any explanation or markdown formatting.
        4. Use aggregate functions (COUNT, SUM, AVG, etc.) appropriately.
        5. Use proper WHERE clauses to filter data.
        6. Add LIMIT clauses for queries that might return many rows (default LIMIT 10 unless user specifies).
        7. For date comparisons, remember dates are stored as TEXT in ISO format.
        """

    async def list_tables(self, ctx: RunContext[DBDeps]) -> List[str]: 
        """Get names of all tables in the database to understand available data."""
        return await ctx.deps.db_service.get_tables()

    async def get_schema(self, ctx: RunContext[DBDeps]) -> SchemaInfo: 
        """Get an overview of the database schema including column and row counts."""
        return await ctx.deps.db_service.get_schema()

    async def describe_table(self, ctx: RunContext[DBDeps], table_name: str) -> TableInfo | None: 
        """Get detailed structure, types, constraints, and relationships for a specific table."""
        return await ctx.deps.db_service.get_table_info(table_name)

    async def explain_query(self, ctx: RunContext[DBDeps], sql_query: str) -> str: 
        """Get the execution plan for a SQL query to understand performance or dependencies."""
        return await ctx.deps.db_service.explain(sql_query)

    async def query(self, ctx: RunContext[DBDeps], sql_query: str) -> QueryResult: 
        """Execute a SQL query on the database to retrieve live data."""
        print(sql_query)
        result = await ctx.deps.db_service.execute(sql_query)
        #required for non-reasoning models(self-correcting)
        if isinstance(result, str):
            raise ModelRetry(
                f"The query failed with an error.\n" 
                f"Original Question: {ctx.prompt}\n"
                f"Failed SQL Query: {sql_query}\n"
                f"Error: {result}\n"
                "Please correct the SQL query and try again."
            )
        return result

    def get_toolset(self) -> AgentToolset[DBDeps] | None: 
        toolset: FunctionToolset[DBDeps] = FunctionToolset()
        toolset.add_function(self.list_tables, name='list_tables')
        toolset.add_function(self.get_schema, name='get_schema')
        toolset.add_function(self.describe_table, name='describe_table')
        toolset.add_function(self.explain_query, name='explain_query')
        toolset.add_function(self.query, name='query', retries=3)
        return toolset
