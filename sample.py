
import collections

# binary search template
def can_solve_with_mid(mid): pass

# Binary Search + Koko eats bananas
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

# Topological Sort

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