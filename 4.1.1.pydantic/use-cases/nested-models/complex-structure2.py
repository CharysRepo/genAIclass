from typing import List
from pydantic import BaseModel, EmailStr, PositiveInt, conlist, Field, HttpUrl

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str


class Employee(BaseModel):
    name: str
    position: str
    email: EmailStr


class Owner(BaseModel):
    name: str
    email: EmailStr


class Restaurant(BaseModel):
    name: str
    owner: Owner
    address: Address
    employees: List[Employee]
    number_of_seats: PositiveInt
    delivery: bool
    website: HttpUrl

restaurant1 = Restaurant(
    name="Tasty Bites",
    owner={"name": "John Doe", "email": "john.doe@example.com"},
    address={
        "street": "123, Flavor Street",
        "city": "Tastytown",
        "state": "TS",
        "zip_code": "12345",
    },
    employees=[
        {"name": "Jane Doe", "position": "Chef", "email": "jane.doe@example.com"},
        {"name": "Mike Roe", "position": "Waiter", "email": "mike.roe@example.com"},
    ],
    number_of_seats=50,
    delivery=True,
    website="http://tastybites.com",
)

print(restaurant1)
print(restaurant1.model_dump())
