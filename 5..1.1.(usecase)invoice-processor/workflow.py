import asyncio
from pathlib import Path
import time
from agents.extractor_agent import extract_invoices_data
from utils.utils import build_invoices_dataframe, generate_financial_summary, create_excel_report

async def workflow():
    base_dir = Path(__file__).parent.resolve() / 'data'
    invoices_dir = base_dir / 'samples'
    temp_dir = base_dir / "temp_files"
    output_dir = base_dir / 'output'
    output_file = output_dir / 'report.xlsx'
    output_dir.mkdir(exist_ok=True)

    invoices_filenames, invoices = await extract_invoices_data(invoices_dir, temp_dir)
    print(invoices)
    invoices_df = build_invoices_dataframe(invoices_filenames, invoices)
    total_s, monthly_df = generate_financial_summary(invoices_df)
    create_excel_report(invoices_df, total_s, monthly_df, output_file)

if __name__ == "__main__":
    start_time = time.perf_counter()
    try:
        asyncio.run(workflow())
    except Exception as e:
        print(e)
    end_time = time.perf_counter() - start_time
    print(f"Invoice Processiong took: {end_time:.1f} seconds")