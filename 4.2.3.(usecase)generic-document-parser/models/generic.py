from pydantic import BaseModel, Field
from typing import Annotated, List

class FileElement(BaseModel):
    """Represents a single element detected on a document page.

    Examples of elements:
    - A table with columns and rows
    - A paragraph of text
    - An image or logo description
    - A header or title
    - A list or bullet points
    """
    element_type: Annotated[
        str, 
        Field(description="Type of the element found on the page", default="")
    ]
    element_content: Annotated[
        str, 
        Field(description="Content of the element found on the page", default="")
    ]


class ModelAnalysisOutput(BaseModel):
    """Structured output from the LLM for document analysis."""

    file_type: Annotated[
        str, 
        Field(
            description="Type name which can describe given file precisely, e.g.: invoice, internal_document, instruction, other", 
            default=""
        )
    ]
    file_elements: Annotated[
        list[FileElement], 
        Field(
            description="Elements the given page consists from: tables, images, paragraphs, graphs, flowcharts etc."
        )
    ]


class OCROutput(BaseModel):
    """Final output combining the filename and structured analysis results."""

    filename: Annotated[str, Field(default="")]
    analysis_result: Annotated[
        ModelAnalysisOutput, 
        Field(default_factory=ModelAnalysisOutput)
    ]
