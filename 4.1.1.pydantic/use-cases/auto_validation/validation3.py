from pydantic import BaseModel, ValidationError

class Product(BaseModel):
    name: str
    quantity: int
    price: float


try:
    external_data = {
        "name": "Laptop",
        "quantity": 3.5,
        "price": 200
    }

    p1 = Product(**external_data)
    print(p1)
except ValidationError as e:
    print(e)
