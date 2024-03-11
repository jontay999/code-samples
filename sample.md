# Used for screenshots in slides

## General

```bash
python3 -m venv <venv-name>

```

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
s1 = "101"
s1.zfill(8) # -> 00000101

s = int(s1, 2)
bin(s)[2:] == s1
```

```py
print(1,2,3, sep="-") # -> 1-2-3
f"Formatted: {1+1}" # -> Formatted: 2
```

### List

```py
arr = [1,2,3,3,4]
for idx, val in enumerate(arr):
    ...

arr.count(3) # -> 2
arr[::-1] # reverses list
sorted(arr, key=lambda x: x % 2) # -> ?
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

## Useful Libraries

```py
"""
bisect_left returns the leftmost place in the sorted list to insert the given element.
bisect_right returns the rightmost place in the sorted list to insert the given element.
"""
import bisect
bisect.bisect_left([1,2,3], 2) # 1
bisect.bisect_right([1,2,3], 2) # 2

```

```py
from collections import Counter

arr = [7,7,8,9,9,9]
c = Counter(arr) # -> {7: 2, 8:1, 9:3}

# n most common
c.most_common(n) # -> e.g. n = 2,  [(9, 3), (7, 2)]

# n least common
c.most_common()[:-n-1:-1]

from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity: int):
        self.arr = OrderedDict()
        self.c = capacity

    def get(self, key: int) -> int:
        if key in self.arr:
            val = self.arr[key]
            del self.arr[key]
            self.arr[key] = val
            return val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.arr:
            del self.arr[key]
        elif len(self.arr) ==  self.c:
            self.arr.popitem(last=False)

        self.arr[key] = value


from collections import defaultdict

# default value for keys will be 0
defaultdict(0)

# useful for graphs
defaultdict(list)



```

## Ethics

```py
import sys
limit = sys.getrecursionlimit()
sys.setrecursionlimit(2000)
```

```py
if fast_check() and slow_check():
    ...

memo = {}
def dfs(node, count):
    if is_valid(node): return
    if count >= max_num: return
    if (node, count) in memo: return
    ...

```

```py
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        """
        case 1: use the curr node + best sum using left child + best sum using right child
        case 2: use curr node + best sum using right child
        case 3: use curr node + best sum using left child
        case 4: best sum of right subtree
        case 5: best sum of left subtree
        """
        best = float('-inf')
        def dfs(node):
            nonlocal best
            if node == None: return 0
            left = max(0,dfs(node.left))
            right = max(0,dfs(node.right))
            best = max(best, left+right+node.val)
            return max(left,right) + node.val
        dfs(root)
        return best
```

```py
def maxPathSum(self, root: Optional[TreeNode]) -> int:
    best = float('-inf')
    d = {}

    # initialize a stack
    s = [root]

    # some helper functions
    done = lambda node: node == None or (id(node) in d)
    get_val = lambda node: max(d.get(id(node), -1), 0)

    while s:
        node = s.pop()

        # check if the current node can be processed yet
        if done(node.left) and done(node.right):
            left = get_val(node.left)
            right = get_val(node.right)
            best = max(best, left + right + node.val)
            d[id(node)] = max(left, right) + node.val
        else:
            # add current node back
            s.append(node)

            # add the other nodes that need to be processed first
            if node.left: s.append(node.left)
            if node.right: s.append(node.right)
    return best
```

```py
def solution(param1, param2):
    if param1 < 0:
        return 1 / 0
    ...
    raise f"Param 1:{param1}, Param2: {param2}"
    return
```
