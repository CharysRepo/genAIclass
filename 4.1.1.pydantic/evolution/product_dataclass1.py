from dataclasses import dataclass

@dataclass
class Product:
    name: str
    quantity: int
    price: float

p1 = Product("Laptop", 3, 200)
print(p1)
p2 = Product("LED", 2, 300)
print(p2)
p3 = Product("Laptop", 3, 200)
print(p3)
print(p1 == p3)
print(p1 == p2)
