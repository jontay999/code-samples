# arrays
# negative indexing, [::-1], reversing, list(map()), lambda sorts
# list(map (int, input().split()))
# find most frequent element
arr = [1,2,3,3,3,2]

# level 1: create dict -> count
max(set(arr), key = arr.count)

# dictionaries

# combining dictionaries
x = {'a': 1, 'b': 2}
y = {'b': 3, 'c': 4}
z = {**x, **y}

# getting value with .get



# Tricks

# get sys recursion limit
    # convert recursive function to stack
    # convert multidimensional dp to 2 row dp