from pydantic import BaseModel, Field
from typing import Annotated, Optional, List

class Address(BaseModel):
    """Represents the geographic location details of a person or organization."""
    street: Annotated[Optional[str], Field(default=None, description="The street address, e.g., 123 Main St")]
    city: Annotated[Optional[str], Field(default=None, description="The city, e.g., Springfield")]
    state: Annotated[Optional[str], Field(default=None, description="The state or province code, e.g., IL")]
    postal_code: Annotated[Optional[str], Field(default=None, description="The postal or ZIP code, e.g., 62704")]
    country: Annotated[Optional[str], Field(default=None, description="The country, e.g., USA")]

class LineItem(BaseModel):
    """Represents an individual product or service entry listed in a document."""
    amount: Annotated[float, Field(description="The total amount for this line item")]
    description: Annotated[Optional[str], Field(default=None, description="Description of the product/service, e.g., Laptop")]
    product_code: Annotated[Optional[str], Field(default=None, description="The SKU or product code, e.g., LPT-001")]
    quantity: Annotated[int, Field(description="The number of units")]
    unit: Annotated[Optional[str], Field(default=None, description="Unit of measure, e.g., pcs")]
    unit_price: Annotated[float, Field(description="Price per single unit")]

class VAT(BaseModel):
    """Represents Value Added Tax details, including rates and calculated amounts."""
    amount: Annotated[float, Field(description="Taxable amount")]
    category_code: Annotated[Optional[str], Field(default=None, description="Tax category code, e.g., A")]
    tax_amount: Annotated[Optional[float], Field(default=None, description="The calculated tax value")]
    tax_rate: Annotated[Optional[float], Field(default=None, description="Percentage as a float, e.g., 10 for 10%")]
    total_amount: Annotated[float, Field(description="Total amount including tax")]

class Party(BaseModel):
    """Represents the contact and identification details of a transacting entity."""
    name: Annotated[str, Field(description="Name of the entity, e.g., Google")]
    street: Annotated[Optional[str], Field(default=None, description="Street address")]
    city: Annotated[Optional[str], Field(default=None, description="City name")]
    state: Annotated[Optional[str], Field(default=None, description="State or province")]
    postal_code: Annotated[Optional[str], Field(default=None, description="Postal code")]
    country: Annotated[Optional[str], Field(default=None, description="Country name")]
    email: Annotated[Optional[str], Field(default=None, description="Contact email address")]
    phone: Annotated[Optional[str], Field(default=None, description="Contact phone number")]
    website: Annotated[Optional[str], Field(default=None, description="Entity website URL")]
    tax_id: Annotated[Optional[str], Field(default=None, description="Tax identification number")]
    registration: Annotated[Optional[str], Field(default=None, description="Business registration number")]
    iban: Annotated[Optional[str], Field(default=None, description="International Bank Account Number")]
    payment_ref: Annotated[Optional[str], Field(default=None, description="Payment reference or invoice number")]

class Invoice(BaseModel):
    """Represents the comprehensive structured data extracted from an invoice document."""
    invoice_id: Annotated[str, Field(description="The unique identifier of the invoice")]
    invoice_date: Annotated[str, Field(description="The date the invoice was issued (YYYY-MM-DD)")]
    supplier: Party
    receiver: Party
    line_items: List[LineItem]
    vat: List[VAT]
