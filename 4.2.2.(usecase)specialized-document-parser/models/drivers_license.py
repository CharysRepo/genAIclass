from pydantic import BaseModel, Field
from datetime import date

from datetime import date
from typing import Annotated
from pydantic import BaseModel, Field

class DriversLicense(BaseModel):
    """
    Captures personal identity and licensing details extracted from a 
    government-issued driver's license.
    """
    
    address: Annotated[
        str, 
        Field(title="Address", description="The address of the individual.")
    ]
    
    date_of_birth: Annotated[
        date, 
        Field(title="Date of Birth", description="The birthdate of the individual.")
    ]
    
    document_id: Annotated[
        str, 
        Field(
            title="Document ID", 
            description="The unique document ID for the driver's license."
        )
    ]
    
    expiration_date: Annotated[
        date, 
        Field(
            title="Expiration Date", 
            description="The expiration date of the driver's license."
        )
    ]
    
    family_name: Annotated[
        str, 
        Field(
            title="Family Name", 
            description="The family name (last name) of the individual."
        )
    ]
    
    given_names: Annotated[
        str, 
        Field(
            title="Given Names", 
            description="The given names (first and middle names) of the individual."
        )
    ]
    
    issue_date: Annotated[
        date, 
        Field(title="Issue Date", description="The issue date of the driver's license.")
    ]
