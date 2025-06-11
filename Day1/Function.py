#Function Definition
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))  # Hello, Alice!
print(greet("Bob", "Hi"))  # Hi, Bob!

# changeable variable
def sum_numbers(*args):
    return sum(args)
print(sum_numbers(1, 2, 3, 4))  # 10

# Anonymous function
double = lambda x: x * 2
print(double(5))  # 10

# high-level function
def apply_func(func, value):
    return func(value)
print(apply_func(lambda x: x ** 2, 4))  # 16
