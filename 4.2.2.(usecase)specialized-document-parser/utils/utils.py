import json
import os
import tempfile
from pathlib import Path
import PyPDF2
from pdf2image import convert_from_path
from models.drivers_license import DriversLicense
from models.invoice import Invoice
from models.payslip import Payslip

def list_all_files(directory: Path, extension: str = "pdf") -> list[Path]:
    return [Path(directory, f) for f in os.listdir(directory) if f.lower().endswith(extension)]

def pdf_to_jpg(file_path: Path, save_dir: Path) -> list[Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file_path).stem

    image_paths = []

    # Use temporary directory to avoid cluttering the save directory with intermediate files
    # convert_from_path returns PIL Image objects, we save them as JPG
    with tempfile.TemporaryDirectory() as path:
        images = convert_from_path(file_path, output_folder=path)

        for idx, image in enumerate(images):
            save_path = Path(save_dir, f"{filename}_page_{idx}.jpg")
            image.save(save_path, "JPEG")
            image_paths.append(save_path)

    return image_paths

def extract_text_from_pdf(file_path: Path) -> str:
    with file_path.open(mode='rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def save_results_to_json(filenames: str, results: list[Invoice | Payslip | DriversLicense], save_dir: Path) -> None:
    for filename, result in zip(filenames, results):
        save_path = Path(save_dir, f"{filename.stem}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump_json(), f, indent=4, ensure_ascii=True)
