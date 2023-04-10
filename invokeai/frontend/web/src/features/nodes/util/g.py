import collections


class Graph(object):
    def __init__(self, edges):
        self.edges = edges
        self.adj = Graph._build_adjacency_list(edges)

    @staticmethod
    def _build_adjacency_list(edges):
        adj = collections.defaultdict(list)
        for edge in edges:
            adj[edge[0]].append(edge[1])
            adj[edge[1]] # side effect only
        return adj


def dfs(G):
    discovered = set()
    finished = set()

    for u in G.adj:
        if u not in discovered and u not in finished:
            discovered, finished = dfs_visit(G, u, discovered, finished)


def dfs_visit(G, u, discovered, finished):
    discovered.add(u)

    for v in G.adj[u]:
        # Detect cycles
        if v in discovered:
            print(f"Cycle detected: found a back edge from {u} to {v}.")
            break

        # Recurse into DFS tree
        if v not in finished:
            dfs_visit(G, v, discovered, finished)

    discovered.remove(u)
    finished.add(u)

    return discovered, finished


if __name__ == "__main__":
    G = Graph([
        ('u', 'v'),
        ('u', 'x'),
        ('v', 'y'),
        ('w', 'y'),
        ('w', 'z'),
        ('x', 'v'),
        ('y', 'x'),
        ('z', 'z')])
    print(G.adj)

    dfs(G)
