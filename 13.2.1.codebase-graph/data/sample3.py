import math  # Added to test module import tracking
from abc import ABC  # Added to test from-style import tracking

class BaseOperation(ABC):  # Added parent class to test inheritance tracking
    """Abstract base class for operations."""
    pass

class Calculator(BaseOperation):  # Modified to inherit from BaseOperation
    def __init__(self, base_value):
        self.value = base_value

    def add(self, amount):
        # Using imported math module to test deep call references
        self.value += math.ceil(amount)
        return self.value

def process_data(data):
    calc = Calculator(10)
    return calc.add(data)
