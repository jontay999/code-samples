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
1 if cond1 else 2 if cond2 else 3

# conditional list comprehensions
[x for x in range(10) if x % 2 == 0]

[a if a else 2 for a in [0,1,0,3]]
```

```py
print(1,2,3, sep="-") # -> 1-2-3
f"Formatted: {1+1}" # -> Formatted: 2
```

## Queues

```py
from collections import deque
q = deque([])
q.append(1)
q.append(2)
q.popleft() # -> 1
q.appendleft(3)
q.popleft() # -> 3
```

## Heaps

```py
from heapq import heapify, heappop, heappush


h = []
heappush(h, 3)
heappush(h, 2)
heappop(h) # -> ?

# Python default is min-heap
# use negative numbers for max-heap



from heapq import nsmallest, nlargest

n_largest_elements = nlargest(n, heap)
n_smallest_elements = nsmallest(n, heap)
```

## Sets

```py
s1 = set()
s1.add(1)
s.add(4)
s.add(5)
s.remove(5)
s.discard(6)
# remove vs discard?

s2 = {i for i in range(3)}

s3 = s1.intersection(s2) # -> ?
s1 | s2 # -> ?
s1 & s2 # -> ?
s1 ^ s3 # -> ?
```

## Dictionaries

```py
d = {k:v for k,v in enumerate(range(4))} # -> ?
d.get(3, -1)  # -> ?

for k,v in d.items():
    ...

del d[1] # -> ?
d.pop(2) # -> ?
```
