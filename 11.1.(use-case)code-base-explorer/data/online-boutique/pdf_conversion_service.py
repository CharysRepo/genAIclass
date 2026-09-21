import asyncio
from pathlib import Path
import tempfile
import pymupdf
from agents.image_to_text_agent import image_to_text_agent
from pydantic_ai import BinaryContent

class PdfToTextConversionService:
    def __init__(self):
        pass
   
    def convert_pdf_to_images(self, pdf_path:Path, img_dir:str):
        zoom_x = 2.0  
        zoom_y = 2.0 
        mat = pymupdf.Matrix(zoom_x, zoom_y) 

        doc = pymupdf.open(pdf_path)
        for page in doc: 
            pix = page.get_pixmap(matrix=mat) 
            save_path = Path(img_dir, f"{pdf_path.stem}_page_{page.number}.jpg")
            pix.save(save_path)

    async def convert_image_to_text(self, image: Path) -> str:
        #print(f"convert_image_to_text{image}")
        prompt = [
            """Extract all content from the image. 
            Format regular text as paragraphs and extract tables 
            maintaining their structure.""",
            BinaryContent(data=image.read_bytes(), media_type="image/jpeg")
        ]
        result = await image_to_text_agent.run(prompt)        
        return result.output
   
    async def convert_pdf_to_text(self, pdf_path: Path) -> str:
        with tempfile.TemporaryDirectory() as temp_path:
            temp_dir = Path(temp_path)
            self.convert_pdf_to_images(pdf_path, img_dir=temp_dir)

            image_files = sorted(temp_dir.glob("*"))
            tasks = [self.convert_image_to_text(file) for file in image_files]        
            results = await asyncio.gather(*tasks)
            formatted_results = "\n\n--- Page Break ---\n\n".join(results)
        return formatted_results
            
