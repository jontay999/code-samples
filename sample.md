# Used for screenshots in slides

```py
float('inf') # max number
float('-inf') # min number
3 ^ 2 # xor
3**2 # squaring
```

```py
# multiple assignment
a,b = 0,0
a = 0; b = 0
```

```py
assert 1 == 1, "Custom message if assert fails"
raise "Custom Error"
```

```py
# splat
a = [1, 2, 3]
print(*a) # --> print(1, 2, 3)
```

```py
global_var = 1
def fn1():
    global global_var
    global_var += 1

    local_var = 2
    def fn2():
        nonlocal local_var
        local_var += 1
```

```py
class Parent:
    int num # attribute

    def __init__():
        pass

    # method
    def method1(): pass

def Child(Parent):
    def method1():
        print("this is how you override")
```

```py
# ternarys
cond1, cond2 = True,False
name = "John" if cond1 else "Doe"
[a if a else 2 for a in [0,1,0,3]]
1 if cond1 else 2 if cond2 else 3
```
