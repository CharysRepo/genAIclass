class Calculator:
    def __init__(self, base_value):
        self.value = base_value

    def add(self, amount):
        self.value += amount
        return self.value

def process_data(data):
    calc = Calculator(10)
    return calc.add(data)