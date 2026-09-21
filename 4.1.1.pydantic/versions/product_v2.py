from typing import Annotated

from pydantic import BaseModel, Field, ValidationError

class Product(BaseModel):
    name: Annotated[str, Field(description="Product name")]
    price: Annotated[float, Field(description="Price of the product")]
    quantity: Annotated[int, Field(description="Available stock quantity")]
    rating: Annotated[int | None, Field(description="User rating between 1-5 stars", ge=1, le=5)] = None
    features: Annotated[list[str], Field(description="Key product features")]

print("--- Testing Valid Data ---")
try:
    p1 = Product(name="Laptop", price=999.99, quantity=10, rating=5, features=["SSD", "16GB RAM"])
except ValidationError as e:
    print(e)

print("\n--- Testing Invalid Rating (6) ---")
try:
    Product(name="Phone", price=500.0, quantity=5, rating=6, features=["5G"])
except ValidationError as e:
    print(e)
