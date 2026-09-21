import asyncio
import asyncpg
from pydantic_evals import Case, Dataset
from tasks.hybrid_search_service import HybridSearchService
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from dotenv import load_dotenv
load_dotenv(override=True)
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

# Out of all the items that are expected to be retrieved, how many were actually retrieved?
class MultiRecall(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext) -> float:
        expected, actual = set(ctx.expected_output), set(ctx.output)
        return len(expected.intersection(actual)) / len(expected) if expected else 0.0

# Out of all the items that were retrieved, how many are actually correct?
class MultiPrecision(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext) -> float:
        expected, actual = set(ctx.expected_output), set(ctx.output)
        return len(actual.intersection(expected)) / len(actual) if actual else 0.0

class F1Score(Evaluator):
    """Harmonic mean of precision and recall."""
    async def evaluate(self, ctx: EvaluatorContext) -> float:
        expected, actual = set(ctx.expected_output), set(ctx.output)
        intersect = len(expected.intersection(actual))
        
        precision = intersect / len(actual) if actual else 0.0
        recall = intersect / len(expected) if expected else 0.0
        
        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

hybrid_search_service = HybridSearchService()

async def hybrid_search_task(query: str) -> list[str]:
    results = await hybrid_search_service.search(query, 10)
    return [r["chunk_id"] for r in results[:3]]

async def run_evaluation(limit: int = 5):
    conn = await asyncpg.connect(dsn="postgresql://postgres:postgres@localhost:5432/postgres")
    
    rows = await conn.fetch(
        "SELECT query, ground_truth_chunk_ids FROM golden_records2 LIMIT $1", 
        limit
    )

    cases = [
        Case(
            name=f"MultiHop_{i}",
            inputs=row["query"],
            expected_output=row["ground_truth_chunk_ids"],
        )
        for i, row in enumerate(rows)
    ]

    dataset = Dataset(
        name="Multi-Hop Search Benchmark",
        cases=cases,
        evaluators=[
            MultiPrecision(),
            MultiRecall(),
            F1Score()
        ]
    )

    print(f"📊 Evaluating {len(cases)} cases with F1-Score...")
    report = await dataset.evaluate(hybrid_search_task)
    report.print(include_input=True, include_expected_output=True, include_output=True)
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(run_evaluation())
