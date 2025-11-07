import math
import random
from itertools import combinations
from typing import List, Tuple

import networkx as nx


def christofides_tsp(G: nx.Graph, weight: str = "weight") -> Tuple[List, float]:
    """
    Compute a TSP tour using Christofides' heuristic on an undirected metric graph G.

    Returns:
        tour: list of nodes in cycle order (first node not repeated at end)
        tour_cost: total cost of the tour
    Preconditions:
        - G is undirected and (preferably) complete with nonnegative edge weights.
        - weight is the edge attribute key for cost.
    """
    # 1. Minimum spanning tree T
    T = nx.minimum_spanning_tree(G, weight=weight)

    # 2. Vertices with odd degree in T
    odd_degree_nodes = [v for v, d in T.degree() if d % 2 == 1]

    # 3. Minimum weight perfect matching on the induced complete graph over odd-degree nodes
    # Build a complete graph on odd-degree nodes using weights from G
    K = nx.Graph()
    K.add_nodes_from(odd_degree_nodes)
    for u, v in combinations(odd_degree_nodes, 2):
        # get weight from G (assume exists; if not, consider using shortest path distances)
        w = G[u][v].get(weight)
        if w is None:
            # fallback to shortest-path distance if direct edge not present
            w = nx.shortest_path_length(G, u, v, weight=weight)
        K.add_edge(u, v, **{weight: w})

    # Try to use min_weight_matching if available; otherwise use max_weight_matching on negated weights
    try:
        matching_edges = nx.algorithms.matching.min_weight_matching(K, maxcardinality=True, weight=weight)
    except Exception:
        # fallback: transform weights and use max_weight_matching
        for u, v, data in K.edges(data=True):
            data["_neg_w"] = -data[weight]
        matching_edges = nx.algorithms.matching.max_weight_matching(K, maxcardinality=True, weight="_neg_w")

    # matching_edges is a set of (u,v) pairs
    # 4. Multigraph H = T + matching edges
    H = nx.MultiGraph()
    H.add_nodes_from(T.nodes())
    H.add_edges_from(T.edges(data=True))
    for u, v in matching_edges:
        w = K[u][v][weight]
        H.add_edge(u, v, **{weight: w})

    # 5. Eulerian circuit of H (H is Eulerian: connected and all degrees even)
    if not nx.is_eulerian(H):
        # If H is not Eulerian, ensure connectedness (rare if original G is complete)
        # This is a safety fallback; usually Christofides guarantees Eulerian.
        H = nx.eulerize(H)  # convert to Eulerian by duplicating edges (networkx has this helper in some versions)
    circuit = list(nx.eulerian_circuit(H))

    # Convert Eulerian circuit edges to a node sequence
    euler_nodes = []
    for edge in circuit:
        # edge could be (u, v) or (u, v, key) depending on networkx version
        u, v = edge[0], edge[1]
        if not euler_nodes:
            euler_nodes.append(u)
        euler_nodes.append(v)

    # 6. Shortcutting to form Hamiltonian cycle
    visited = set()
    tour = []
    for node in euler_nodes:
        if node not in visited:
            tour.append(node)
            visited.add(node)
    # Ensure it's a cycle: tour returns to start implicitly when computing cost
    # Compute total cost
    def path_cost(path: List) -> float:
        total = 0.0
        n = len(path)
        for i in range(n):
            u = path[i]
            v = path[(i + 1) % n]
            total += G[u][v][weight]
        return total

    tour_cost = path_cost(tour)
    return tour, tour_cost


# ---------------------
# Example usage: random Euclidean TSP
# ---------------------
def euclidean_complete_graph(points: List[Tuple[float, float]]) -> nx.Graph:
    """Build a complete undirected graph with Euclidean distances as weights."""
    G = nx.Graph()
    n = len(points)
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            (x1, y1), (x2, y2) = points[i], points[j]
            d = math.hypot(x1 - x2, y1 - y2)
            G.add_edge(i, j, weight=d)
    return G


if __name__ == "__main__":
    random.seed(0)
    n = 12  # change for more nodes
    pts = [(random.random(), random.random()) for _ in range(n)]
    G = euclidean_complete_graph(pts)

    tour, cost = christofides_tsp(G)
    print("Tour nodes:", tour)
    print("Tour cost: {:.6f}".format(cost))

    # Sanity: show simple 2-approx (MST doubling) cost for comparison
    T = nx.minimum_spanning_tree(G)
    doubled = list(nx.eulerian_circuit(nx.MultiGraph(T).copy()))
    nodes_double = []
    for e in doubled:
        u, v = e[0], e[1]
        if not nodes_double:
            nodes_double.append(u)
        nodes_double.append(v)
    visited = []
    seen = set()
    for node in nodes_double:
        if node not in seen:
            visited.append(node)
            seen.add(node)
    two_approx_cost = sum(G[visited[i]][visited[(i + 1) % len(visited)]]['weight'] for i in range(len(visited)))
    print("MST-doubling (2-approx) cost: {:.6f}".format(two_approx_cost))

