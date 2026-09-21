from pydantic import BaseModel, ValidationError

class Product(BaseModel):
    name: str
    quantity: int
    price: float


try:
    p1 = Product(name="Laptop", quantity=3.5, price=200)
    print(p1)
    p2 = Product(name="LED", quantity=2, price=300)
    print(p2)
    p3 = Product(name="Laptop", quantity=3, price=200)
    print(p3)
    print(p1 == p3)
    print(p1 == p2)
except ValidationError as e:
    print(e)
