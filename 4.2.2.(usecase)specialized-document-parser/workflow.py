import asyncio
from pathlib import Path
import time
from agents.specialized_doc_agents import run_inference   
from utils.utils import list_all_files, pdf_to_jpg, save_results_to_json
from tqdm.asyncio import tqdm_asyncio

async def extract_structure_docs(pdf_dir:Path, temp_dir:Path):
    filenames = []
    tasks = []
    pdf_paths = list_all_files(pdf_dir)
    for pdf_path in pdf_paths:
        print(pdf_path)
        filenames.append(pdf_path)
        image_paths = pdf_to_jpg(pdf_path, temp_dir)
        task = asyncio.create_task(run_inference(image_paths[0]))
        tasks.append(task)
    invoices_result = await tqdm_asyncio.gather(*tasks)
    return filenames, invoices_result 

async def workflow():
    base_dir = Path(__file__).parent.resolve() / "data" 
    pdf_dir = base_dir / "samples"
    temp_dir = base_dir / "temp_files"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("### Parsing Documents Begin")
    filenames, results = await extract_structure_docs(pdf_dir, temp_dir)
    print("### Parsing Documents End")
    print("### Saving results to json output files")
    save_results_to_json(filenames, results, results_dir)


if __name__ == "__main__":
    start_time = time.perf_counter()
    try:
        asyncio.run(workflow())
    except Exception as e:
        print(e)
    end_time = time.perf_counter() - start_time
    print(f"Analysis took: {end_time:.1f} seconds")