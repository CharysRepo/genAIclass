class Product:
    def __init__(self, name: str, quantity: int, price: float):
        print("Product.__init__ called")
        self.name = name
        self.quantity = quantity
        self.price = price

    def __repr__(self):
        print("Product.__repr__ called")
        return self.name + " " + str(self.quantity) + " " + str(self.price)

    def __eq__(self, p):
        print("Product.__eq__ called")
        return (
            self.price == p.price
            and self.quantity == p.quantity
            and self.name == p.name
        )


p1 = Product("Laptop", 3, 200)
print(p1)
p2 = Product("LED", 2, 300)
print(p2)
p3 = Product("Laptop", 3, 200)
print(p3)
print(p1 == p3)
print(p1 == p2)