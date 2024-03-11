
import collections
from heapq import heappush, heappop, heapify

def can_solve_with_mid(mid): pass

# Binary Search Template -> Koko eats bananas
def bin_search(arr):
    lo, hi = 0, len(arr)
    ans = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if can_solve_with_mid(mid):
            ans = mid
            lo = mid + 1
        else:
            hi = mid-1
    return ans

# 1D dp -> House Robber

# 2D dp -> Edit distance

# Topological Sort -> Courses
def topo_sort(edges, n):
    in_deg = [0] * n
    g = {i: set() for i in range(n)}
    for x,y in edges:
        g[x].add(y)
        in_deg[y] += 1

    q = collections.deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    while q:
        node = q.popleft()
        topo.append(node)
        for neigh in g[node]:
            in_deg[neigh] -= 1
            if in_deg[neigh] == 0:
                q.append(neigh)

# Kosaraju
def KosarajuExample():
    """
    Run DFS on graph, populate stack
    Reverse Graph
    Run DFS on reversed graph, popping nodes of the stack
    """
    g = {...}

    def dfs(g,node, visited, stack):
        ...

    def dfs2(g, node, visited, scc):
        ...


    def kosaraju(g):
        sccs = []
        stack = []
        visited = [False] * len(g)
        for i in g:
            if visited[i]: continue
            dfs(g,i,visited, stack)

        reversed_g = {i: [] for i in g}
        for x, ys in g.items():
            for y in ys: reversed_g[y].append(x)

        visited = [False] * len(g)
        while stack:
            node = stack.pop()
            if visited[node]: continue
            scc = []
            dfs2(reversed_g, node, visited, scc)

# union find
def UnionFind(n):
    parents = [i for i in range(n)]
    rank = [1] * n # or size
    
    def find(x):
        if parents[x] != x: parents[x] = find(parents[x])
        return parents[x]
    
    def union(x,y):
        xroot, yroot = find(x),find(y)
        if xroot == yroot: return 
        parents[xroot] = yroot
    
    def union_by_rank(x,y):
        xroot, yroot = find(x),find(y)
        if xroot == yroot: return 
        if rank[xroot] < rank[yroot]:
            xroot, yroot = yroot, xroot
        parents[yroot] = xroot
        rank[xroot] = max(rank[xroot], rank[yroot]+1)


# network delay time
def BellmanFord(n, source):
    edges = [...]
    dist = {i: float('inf') for i in range(n)}
    dist[source] = 0
    for _ in range(n):
        for s,t, weight in edges:
            dist[t] = min(dist[t], dist[s] + weight)
    
# kruskals/prims -> Minimum spanning tree
def Prims(g):
    # where g[a] = [(weight, neighbor)...]
    v, h, edges = {0},[],[]
    for w,b in g[0]:
        heappush(h, (w,b,0))

    while len(v) != len(g):
        w,b,a = heappop(h)
        if b in v: continue
        v.add(b)
        edges.append((w,b,a))
        for w2, b2 in g[b]:
            if b2 in v: continue
            heappush(h, (w2,b2, b))
    total_weight = sum(w for w,b,a in edges)
    return edges, total_weight

# Segment Trees
# answers queries of range sums from index i to j
# supports updates
class NumArray:
    def __init__(self, nums):
        self.arr = [0] * len(nums)
        self.BIT = [0] * (len(nums) + 1)
        for i, n in enumerate(nums): self.update(i, n)
        self.sumRange = lambda i, j: self.Sum(j + 1) - self.Sum(i)

    def update(self, i, val):
        diff, self.arr[i] = val - self.arr[i], val
        i += 1
        while i < len(self.BIT):
            self.BIT[i] += diff
            i += (i & -i) # to next

    def Sum(self, k):
        res = 0
        while k:
            res += self.BIT[k]
            k -= (k & -k) # to parent
        return res


# dijkstra
# TBD
            
# bfs shortest path
def BFS():
    start, end = 0, 10
    graph = {}
    q = collections.deque([start])
    visited = {start}
    count = 0
    while q:
        c = len(q)
        for _ in range(c):
            node = q.popleft()
            for neigh in graph[node]:
                if neigh in visited: continue
                q.append(neigh)
        c += 1
    return count



# monotonic stack
# e.g. find idx of the next greatest element
def mono_stack(arr):
    # idx of next element greater than current value
    next_greater = [0 for _ in arr]
    stack = []

    for i in range(len(arr)):
        while stack and arr[stack[-1]] < arr[i]:
            top = stack.pop()
            next_greater[top] = i
        stack.append(i)
    return next_greater


# Bit Tricks
def bit_tricks(num):
    set_bit_pos, unset_bit_pos, toggle_bit_pos, check_bit_pos = 1,1,1
    # all 0-indexed

    # set bit
    num |= 1<<set_bit_pos

    # unset bit
    num &= ~(1 << unset_bit_pos)

    # check bit
    is_set = (num >> check_bit_pos) & 1


    #toggle bit
    num ^= 1 << toggle_bit_pos


    # remove last set bit
    i &= (i-1)

    # Count number of set bits
    count = 0
    number = 10
    while number:
        number &= number-1
        count += 1