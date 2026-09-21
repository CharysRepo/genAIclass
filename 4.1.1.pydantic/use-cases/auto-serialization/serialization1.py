from pydantic import BaseModel

class Product(BaseModel):
    name: str
    quantity: int
    price: float

p1 = Product(name="Laptop", quantity=3, price=200)
print(p1)
print(p1.model_dump())
print(p1.model_dump_json()) # content serialization
print(p1.model_json_schema()) # structure serialization
