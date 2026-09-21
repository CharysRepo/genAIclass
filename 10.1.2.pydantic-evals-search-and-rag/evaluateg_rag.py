import asyncio
import asyncpg
from pydantic_evals import Case, Dataset
from tasks.hybrid_search_service import HybridSearchService
from tasks.rag_agent import KnowledgeDeps, kb_agent
from pydantic_evals.evaluators import LLMJudge
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

deps = KnowledgeDeps(hybrid_search_service=HybridSearchService())

async def rag_task(query: str) -> str:
    result = await kb_agent.run(query, deps=deps)
    return result.output

async def run_evaluation(limit: int = 1):
    conn = await asyncpg.connect(dsn="postgresql://postgres:postgres@localhost:5432/postgres")
    
    # Fetch data: query, reference answer, and the actual text of the source chunks
    rows = await conn.fetch("""
        SELECT 
            g.query, 
            g.expected_answer, 
            (SELECT string_agg(chunk_content, ' | ') 
             FROM document_chunks 
             WHERE chunk_id = ANY(g.ground_truth_chunk_ids)) as source_context
        FROM golden_records2 g
        LIMIT $1
    """, limit)

    cases = [
        Case(
            name=f"RAG_Quality_Case_{i}",
            # Pass both the question and the "Gold" context to the judge
            inputs=f"User Query: {row['query']}\n\nGold Context: {row['source_context']}",
            expected_output=row["expected_answer"],
        )
        for i, row in enumerate(rows)
    ]

    dataset = Dataset(
        name="RAG Quality Benchmark",
        cases=cases,
        evaluators=[
            # 1. Groundedness: Checks for Hallucinations
            LLMJudge(
                model="ollama:glm-4.7-flash:q4_K_M",
                rubric=(
                    "Does the response contain ONLY information found in the 'Gold Context'? "
                    "Score 1.0 if it is perfectly grounded. "
                    "Score 0.0 if it introduces outside information or hallucinations."
                ),
                include_input=False,
                include_expected_output=False,
                score={'evaluation_name': 'groundedness'},
            ),
            # 2. Completeness: Checks for missing information
            LLMJudge(
                model="ollama:glm-4.7-flash:q4_K_M",
                rubric=(
                    "Does the response address ALL relevant facts mentioned in the 'Gold Context' "
                    "that help answer the User Query? "
                    "Score 1.0 if it is a comprehensive answer. "
                    "Score 0.0 if it omits key details from the source context."
                ),
                include_input=True,
                include_expected_output=True, # Use gold answer as a hint for key facts
                score={'evaluation_name': 'completeness'},
            ),
            # 3. Semantic Relevance: Checks for relevance to the query
            LLMJudge(
                model="ollama:glm-4.7-flash:q4_K_M",
                rubric=(
                    "Does the output matches semantically with expected output?"
                    "Score 1.0 if it is a comprehensive answer. "
                    "Score 0.0 if it omits key details from the source context."
                ),
                include_input=False,
                include_expected_output=True, # Use gold answer as a hint for key facts
                score={'evaluation_name': 'relevance'},
            )

        ]
    )

    print(f"📊 Evaluating {len(cases)} cases for Groundedness & Completeness...")
    report = await dataset.evaluate(rag_task)
    report.print(include_input=True, include_expected_output=True, include_output=True)
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(run_evaluation())
