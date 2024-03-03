from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

def dfs(graph, node, visited):
    if node not in visited:
        print(node)
        visited.add(node)
        for neighbor in graph[node]:
            dfs(graph, neighbor, visited)

def bfs(graph, start):
    visited = set()
    queue = [start]

    while queue:
        node = queue.pop(0)
        if node not in visited:
            print(node)
            visited.add(node)
            queue.extend(graph[node])

g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 3)
g.add_edge(1, 2)
g.add_edge(1, 5)
g.add_edge(1, 6)
g.add_edge(2, 3)
g.add_edge(2, 4)
g.add_edge(2, 5)
g.add_edge(3, 2)
g.add_edge(3, 4)
g.add_edge(4, 2)
g.add_edge(4, 6)


print('DFS:')
dfs(g.graph, 0, set())

print('BFS:')
bfs(g.graph, 0)
