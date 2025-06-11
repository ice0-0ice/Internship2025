# Variable Type
name = "Alice"  # str
age = 20        # int
grades = [90, 85, 88]  # list
info = {"name": "Alice", "age": 20}  # dict

# Type change
age_str = str(age)
number = int("123")

# function zone
x = 10
def my_function():
    y = 5
    global x
    x += 1
    print(f"Inside function: x={x}, y={y}")

my_function()
print(f"Outside function: x={x}")
