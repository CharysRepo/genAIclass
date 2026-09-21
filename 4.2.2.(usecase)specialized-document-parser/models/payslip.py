from datetime import date
from typing import Annotated, Optional
from pydantic import BaseModel, Field

class Payslip(BaseModel):
    """Represents the structured payroll, tax withholding, and employment details extracted from a payslip."""

    employee_id: Annotated[str, Field(description="Unique identifier for the employee")]
    employee_name: Annotated[str, Field(description="Full name of the employee")]
    
    pay_period_start: Annotated[date, Field(description="Start date of the pay period")]
    pay_period_end: Annotated[date, Field(description="End date of the pay period")]
    
    gross_income: Annotated[float, Field(description="Total income before deductions")]
    federal_tax: Annotated[float, Field(description="Federal tax deduction amount")]
    state_tax: Annotated[Optional[float], Field(default=0.0, description="State tax deduction amount, if applicable")]
    
    social_security: Annotated[float, Field(description="Social Security deduction amount")]
    medicare: Annotated[float, Field(description="Medicare deduction amount")]
    other_deductions: Annotated[Optional[float], Field(default=0.0, description="Other deductions (e.g., health insurance, retirement)")]
    
    net_income: Annotated[float, Field(description="Income after all deductions")]
    payment_date: Annotated[date, Field(description="Date the payment was issued")]
    
    hours_worked: Annotated[Optional[float], Field(default=None, description="Total hours worked in the pay period")]
    hourly_rate: Annotated[Optional[float], Field(default=None, description="Employee's hourly rate, if applicable")]