from pydantic import BaseModel

class Product(BaseModel):
    name: str
    quantity: int
    price: float

external_data = {
    "name": "Laptop",
    "quantity": 3,
    "price": 200
}

p1 = Product(**external_data)
print(p1)
print(p1.model_dump())
print(p1.model_dump_json())
print(p1.model_json_schema())
