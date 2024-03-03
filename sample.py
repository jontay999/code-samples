
# binary search template
def special_condition(i): pass

# Vanilla Binary Search
def bin_search(arr):
    lo, hi = 0, len(arr)
    ans = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if special_condition(mid):
            ans = mid
            lo = mid + 1
        else:
            hi = mid-1
    return ans

