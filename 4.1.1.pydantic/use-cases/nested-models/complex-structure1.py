from typing import List, Optional
from pydantic import BaseModel

class Food(BaseModel):
    name: str
    price: float
    ingredients: Optional[List[str]] = None


class Restaurant(BaseModel):
    name: str
    location: str
    foods: List[Food]


restaurant1 = Restaurant(
    name="Tasty Bites",
    location="123, Flavor Street",
    foods=[
        {
            "name": "Cheese Pizza",
            "price": 12.50,
            "ingredients": ["Cheese", "Tomato Sauce", "Dough"],
        },
        {"name": "Veggie Burger", "price": 8.99},
    ],
)

print(restaurant1)
print(restaurant1.model_dump())
