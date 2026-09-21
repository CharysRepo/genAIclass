from typing import Annotated, Literal
from pydantic import BaseModel, EmailStr, Field

class Ticket(BaseModel):
    """model for support tickets."""
    customer_id: Annotated[str, Field(description="Unique identifier for the customer")]
    customer_email: Annotated[EmailStr, Field(description="Contact email of the customer")]
    description: Annotated[str, Field(description="Detailed summary of the issue")]
    priority: Annotated[
        Literal["high", "medium", "low"], 
        Field(description="The priority level of the ticket. If sentiment is 'angry', priority MUST be HIGH regardless of the issue. If sentiment is 'frustrated', priority is at least MEDIUM. Otherwise, determine priority based on issue details.")
    ]
    sentiment: Annotated[Literal["angry", "frustrated", "neutral", "happy"],
                Field(description="The category of the ticket. Return 'angry' if they use caps/insults, 'frustrated' for impatience, 'happy' for satisfaction, or 'neutral' otherwise.")
    ]
    ticket_type: Annotated[Literal["billing", "connection"], Field(description="The type of the ticket")]


class EmailDetails(BaseModel):
    to_email: str
    subject: str
    body: str
