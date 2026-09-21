from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

def task(inp: str) -> str:
    return inp.title()

dataset = Dataset(
    name='title_case',
    cases=[
        Case(name='Case1', inputs='hello', expected_output='Hello'),
        Case(name='Case2', inputs='WORLD', expected_output='World'),
        Case(name='Case3', inputs='pYThon', expected_output='Python'),
    ],
    evaluators=[EqualsExpected()], 
)

if __name__ == '__main__':
    results = dataset.evaluate_sync(task)
    results.print(include_expected_output=True, include_output=True, include_input=True)
