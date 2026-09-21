import json
from typing import Any
from pydantic_ai import AgentToolset, FunctionToolset, RunContext
from pydantic_ai.capabilities import AbstractCapability
from dataclasses import dataclass
from pydantic_ai._instructions import AgentInstructions
from services.neo4j_db_service import AsyncNeo4jDBService

@dataclass
class DBDeps:
    db_service: AsyncNeo4jDBService

@dataclass 
class Neo4jCapability(AbstractCapability[Any]): 

    def get_instructions(self) -> AgentInstructions[Any] | None:
        return """You are a helpful assistant for querying a Neo4j database. 
        While converting natural language questions to valid Neo4j cypher queries, follow these guidelines:
        1. Use only the nodes and relationships mentioned in the schema.
        2. Return ONLY the CYPHER query without any explanation or markdown formatting.
        """

    async def get_schema(self, ctx:RunContext[Any]) -> str:
        """Get an overview of the database schema including nodes, properties and relationships."""

        return """
        # CODEBASE KNOWLEDGE GRAPH SPECIFICATION

        ## SECTION 1: GRAPH DATABASE SCHEMA

        ### Node Types & Properties
        - Module: Represents an imported module or file.
            - Properties: id, name, total_lines
            - Properties: id, name, total_lines
        - GlobalVariable: Represents a variable scoped globally or at the file level.
            - Properties: id, name
        - LocalVariable: Represents a variable scoped within a function or block.
            - Properties: id, name
        - ClassVariable: Represents a variable scoped to a class definition.
            - Properties: id, name
        - AsyncFunction: Dynamic node discovered in live graph database.
            - Properties: id, name, decorators, arguments_list, argument_count, is_nested_function, is_async, docstring, code
        - Async: Represents an asynchrous function or method definition
            - Properties: id, name, decorators, arguments_list, code, argument_count, is_nested_function, is_async, docstring, belongs_to
        - Callable: Represents Function or AsyncFunction or 
            - Properties: id, name, decorators, code, arguments_list, argument_count, is_nested_function, is_async, docstring, belongs_to
        - Class: Represents a class definition
            - Properties: id, name, extends, decorators, instance_attributes, is_nested, code
        - Method: Represents a class method
            - Properties: id, name, decorators, arguments_list, argument_count, is_nested_function, belongs_to, is_async, code
        - AsyncMethod: Dynamic node discovered in live graph database.
            - Properties: id, name, decorators, arguments_list, argument_count, is_nested_function, belongs_to, is_async, code
        
        ### Relationship Types & Topology

        - DEFINES
            - Direction: (Class)-[:DEFINES]->(Method)
            - Description: Indicates that a class defines a method or property.
        - DEFINES_VARIABLE
            - Direction: (Class|Function|Method|Module)-[:DEFINES]->(ClassVariable|GlobalVariable|LocalVariable)
            - Description: Indicates that a class defines a method or property.
        - CALLS
            - Direction: (Function|Method)-[:CALLS]->(Function|Method)
            - Description: Indicates a function call of local modules.
        - CALLS_EXTERNAL
            - Direction: (Function|Method)-[:CALLS]->(Function|Method)
            - Description: Indicates a functiona call of imported modules.
        - EXTENDS
            - Direction: (Class)-[:EXTENDS]->(Class)
            - Description: Indicates a class inheritance or subclass relationship.
        - IMPORTS
            - Direction: (Module|File)-[:IMPORTS]->(Module|File)
            - Description: Indicates what modules/files are imported (source file or external package).

        ## SECTION 2: CYPHER QUERY REFERENCE EXAMPLES

        1. Find a function with a specific name:
        ```cypher
        MATCH (f:Function)
        WHERE f.name = "process_data"
        RETURN f
        ```

        2. Find all functions that call a specific function:
        ```cypher
        MATCH (caller)-[:CALLS]->(callee:Function)
        WHERE callee.name = "process_data"
        RETURN caller
        ```

        3. Find all classes that inherit from a specific class:
        ```cypher
        MATCH (sub:Class)-[:EXTENDS]->(super:Class)
        WHERE super.name = "BaseProcessor"
        RETURN sub
        ```

        4. Find a file and all functions contained within it:
        ```cypher
        MATCH (file:File)-[:DEFINES]->(func:Function)
        WHERE file.path = "src/main.py"
        RETURN func
        ```

        5. Find all files that import a specific module:
        ```cypher
        MATCH (file:File)-[:IMPORTS]->(module:Module)
        WHERE module.name = "pandas"
        RETURN file
        ```
    """
            
    async def execute_cypher_query(self, ctx:RunContext[Any], query: str) -> str:
        """
        Executes a raw Cypher query directly against the graph database.
        Allows for flexible, complex, and arbitrary structural code graph traversal.
        """
        try:
            print(query)
            results = await ctx.deps.db_service.execute_cypher(query)
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            print(f"Error in cypher query execution: {e}")
            return json.dumps({"error": str(e)})
        
    def get_toolset(self) -> AgentToolset[Any] | None: 
        toolset: FunctionToolset[Any] = FunctionToolset()
        toolset.add_function(self.get_schema, name="get_schema")
        toolset.add_function(self.execute_cypher_query, name="execute_cypher_query", retries=3)
        return toolset
