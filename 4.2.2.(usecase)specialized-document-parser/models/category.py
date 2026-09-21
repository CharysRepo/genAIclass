from typing import Annotated, Literal
from pydantic import BaseModel, Field


class DocumentCategory(BaseModel):
    """Specifies the supported classifications for financial and identification documents."""  
    category: Annotated[
        Literal["invoice", "payslip", "driver_license"],
        Field(description="The category of the ticket")
    ]  