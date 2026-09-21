from typing import Annotated, Literal
from pydantic import BaseModel, Field
from datetime import date

class InvoiceItem(BaseModel):
    description: Annotated[str, Field(description="Item description")]
    total: Annotated[float, Field(gt=0, description="Line item total")]

class Company(BaseModel):
    name: Annotated[str, Field(min_length=1)]
    address: Annotated[str, Field(description="Full business address")]
    phone: Annotated[str | None, Field(default=None, pattern=r"^\+?[1-9]\d{1,14}$")]
    email: Annotated[str | None, Field(default=None, examples=["info@company.com"])]

class Invoice(BaseModel):
    invoice_number: Annotated[str, Field(description="invoice number")]
    invoice_date: Annotated[date, Field(description="invoice date")]
    invoice_type: Annotated[Literal["incoming","outgoing"] | None, 
                Field(description=
            """ Set it to 'incoming' if the specified company is the recipient of the invoice.
                Set it to 'outgoing' if the specified company is the issuer of the invoice.
                Set it to None if the specified company does not appear as issuer or recipient.""", 
    default=None)]
    issuer: Annotated[Company, Field(description="The sending entity")]
    recipient: Annotated[Company, Field(description="The receiving entity")]
    invoice_items: Annotated[list[InvoiceItem], Field(min_length=1)]
    subtotal: Annotated[float, Field(ge=0)]
    tax_rate: Annotated[float, Field(default=0, ge=0)]
    tax: Annotated[float, Field(default=0, ge=0)]
    total: Annotated[float, Field(ge=0)]
    terms: Annotated[str | None, Field(default=None)]