import random
import math
import time
from collections import deque

# ---------------------------
# Graph utilities
# ---------------------------
class Graph:
    def __init__(self, edges=None, vertices=None):
        # edges: iterable of (u,v) (undirected assumed)
        self.adj = {}
        if vertices:
            for v in vertices:
                self.adj.setdefault(v, set())
        if edges:
            for u, v in edges:
                self.add_edge(u, v)

    def add_edge(self, u, v):
        self.adj.setdefault(u, set()).add(v)
        self.adj.setdefault(v, set()).add(u)

    def neighbors(self, v):
        return self.adj.get(v, set())

    def vertices(self):
        return list(self.adj.keys())

# ---------------------------
# State / scoring
# ---------------------------
def cycle_score(perm, graph):
    """Number of consecutive edges present in perm (including last->first).
       Max score = len(perm)."""
    n = len(perm)
    if n == 0:
        return 0
    count = 0
    for i in range(n):
        a = perm[i]
        b = perm[(i+1) % n]
        if b in graph.neighbors(a):
            count += 1
    return count

def is_hamiltonian(perm, graph):
    return cycle_score(perm, graph) == len(perm)

# ---------------------------
# Neighbor generators
# ---------------------------
def all_2opt_neighbors(perm):
    """Generator of all 2-opt neighbors: reverse segment [i..j] for 0<=i<j<n."""
    n = len(perm)
    for i in range(n):
        for j in range(i+1, n):
            new = perm[:i] + list(reversed(perm[i:j+1])) + perm[j+1:]
            yield new, (i, j)

def random_2opt_neighbor(perm):
    n = len(perm)
    i = random.randrange(n)
    j = random.randrange(n)
    if i == j:
        return perm[:], (i,j)
    if i > j:
        i, j = j, i
    new = perm[:i] + list(reversed(perm[i:j+1])) + perm[j+1:]
    return new, (i, j)

def random_swap_neighbor(perm):
    n = len(perm)
    i, j = random.sample(range(n), 2)
    new = perm[:]
    new[i], new[j] = new[j], new[i]
    return new, (i, j)

# ---------------------------
# Hill Climbing (with restarts)
# ---------------------------
def hill_climb_from(perm, graph, neighbor_mode='2opt', exhaustive=True, max_no_improve=1_000):
    """
    Steepest-ascent hill climbing starting from perm.
    neighbor_mode: '2opt' or 'swap'
    exhaustive: if True, examine all neighbors at each step; else sample random neighbors.
    Returns best_perm, best_score
    """
    current = perm[:]
    n = len(current)
    current_score = cycle_score(current, graph)
    iters_since_improve = 0

    while True:
        best_neighbor = None
        best_score = current_score
        if neighbor_mode == '2opt':
            if exhaustive and n <= 200:  # avoid huge enumeration for big n
                for neigh, move in all_2opt_neighbors(current):
                    s = cycle_score(neigh, graph)
                    if s > best_score:
                        best_score = s
                        best_neighbor = (neigh, move)
            else:
                # sample neighbors
                for _ in range(min(500, n*(n-1)//2, 1000)):
                    neigh, move = random_2opt_neighbor(current)
                    s = cycle_score(neigh, graph)
                    if s > best_score:
                        best_score = s
                        best_neighbor = (neigh, move)
        else:  # swap
            if exhaustive and n <= 200:
                for i in range(n):
                    for j in range(i+1, n):
                        neigh = current[:]
                        neigh[i], neigh[j] = neigh[j], neigh[i]
                        s = cycle_score(neigh, graph)
                        if s > best_score:
                            best_score = s
                            best_neighbor = (neigh, (i,j))
            else:
                for _ in range(min(1000, n*(n-1)//2)):
                    neigh, move = random_swap_neighbor(current)
                    s = cycle_score(neigh, graph)
                    if s > best_score:
                        best_score = s
                        best_neighbor = (neigh, move)

        if best_neighbor and best_score > current_score:
            current, move = best_neighbor
            current_score = best_score
            iters_since_improve = 0
            # continue climbing
        else:
            break

        iters_since_improve += 1
        if iters_since_improve > max_no_improve:
            break

    return current, current_score

def random_restarts_hill_climb(graph, restarts=50, time_limit=None, neighbor_mode='2opt'):
    """Run hill climb from several random starts and return best found solution."""
    vertices = graph.vertices()
    n = len(vertices)
    best_perm = None
    best_score = -1
    start_time = time.time()

    for r in range(restarts):
        if time_limit and time.time() - start_time > time_limit:
            break
        perm = vertices[:]
        random.shuffle(perm)
        perm, s = hill_climb_from(perm, graph, neighbor_mode=neighbor_mode)
        if s > best_score:
            best_score = s
            best_perm = perm[:]
            if best_score == n:
                break
    return best_perm, best_score

# ---------------------------
# Simulated Annealing
# ---------------------------
def simulated_annealing(graph, initial_perm=None, T0=1.0, cooling_rate=0.995, max_iter=20000):
    """Randomized search with probabilistic uphill moves. Returns best found perm and score."""
    vertices = graph.vertices()
    n = len(vertices)
    if initial_perm is None:
        current = vertices[:]
        random.shuffle(current)
    else:
        current = initial_perm[:]

    current_score = cycle_score(current, graph)
    best_perm = current[:]
    best_score = current_score
    T = T0

    for it in range(max_iter):
        # propose neighbor via random 2-opt
        neighbor, move = random_2opt_neighbor(current)
        neigh_score = cycle_score(neighbor, graph)
        delta = neigh_score - current_score

        if delta >= 0:
            current = neighbor
            current_score = neigh_score
        else:
            # accept with prob exp(delta / T) (delta negative)
            if T > 1e-12 and random.random() < math.exp(delta / T):
                current = neighbor
                current_score = neigh_score

        if current_score > best_score:
            best_score = current_score
            best_perm = current[:]
            if best_score == n:
                break

        # cooling schedule
        T *= cooling_rate
        if T < 1e-12:
            T = 1e-12

    return best_perm, best_score

# ---------------------------
# Tabu Search (swap-based)
# ---------------------------
def tabu_search(graph, tenure=50, max_iter=5000):
    vertices = graph.vertices()
    n = len(vertices)
    current = vertices[:]
    random.shuffle(current)
    current_score = cycle_score(current, graph)
    best_perm = current[:]
    best_score = current_score
    tabu = deque()      # list of recent moves (i,j)
    tabu_set = set()

    for it in range(max_iter):
        # examine candidate swaps and select best non-tabu
        best_candidate = None
        best_candidate_score = -1
        best_move = None

        # sample swaps if n large
        samples = min(1000, n*(n-1)//2)
        tried = set()
        for _ in range(samples):
            i, j = random.sample(range(n), 2)
            key = (min(i,j), max(i,j))
            if key in tried:
                continue
            tried.add(key)
            cand = current[:]
            cand[i], cand[j] = cand[j], cand[i]
            s = cycle_score(cand, graph)
            if key in tabu_set:
                # aspiration: if candidate beats best overall, allow it
                if s > best_score and s > best_candidate_score:
                    best_candidate = cand
                    best_candidate_score = s
                    best_move = key
            else:
                if s > best_candidate_score:
                    best_candidate = cand
                    best_candidate_score = s
                    best_move = key

        if best_candidate is None:
            break

        # make move
        current = best_candidate
        current_score = best_candidate_score

        # update tabu
        tabu.append(best_move)
        tabu_set.add(best_move)
        if len(tabu) > tenure:
            old = tabu.popleft()
            tabu_set.remove(old)

        if current_score > best_score:
            best_score = current_score
            best_perm = current[:]
            if best_score == n:
                break

    return best_perm, best_score

# ---------------------------
# Helper: print solution nicely
# ---------------------------
def print_solution(perm, graph):
    if perm is None:
        print("No permutation provided.")
        return
    n = len(perm)
    sc = cycle_score(perm, graph)
    print(f"Score: {sc}/{n}")
    print("Cycle (as permutation):", perm)
    if sc == n:
        print("This is a valid Hamiltonian cycle.")
    else:
        print("Not a Hamiltonian cycle. Missing edges for some consecutive pairs:")
        for i in range(n):
            a = perm[i]; b = perm[(i+1)%n]
            ok = b in graph.neighbors(a)
            print(f" {a} -> {b}: {'OK' if ok else 'MISSING'}")

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Example 1: simple 4-cycle (square) -> Hamiltonian exists
    edges = [(1,2),(2,3),(3,4),(4,1)]
    g1 = Graph(edges=edges)
    print("Square graph")
    perm, score = random_restarts_hill_climb(g1, restarts=20)
    print_solution(perm, g1)
    # SA
    perm_sa, score_sa = simulated_annealing(g1, T0=1.0, cooling_rate=0.995, max_iter=2000)
    print("\nSimulated Annealing result:")
    print_solution(perm_sa, g1)

    # Example 2: star graph -> no Hamiltonian cycle
    center = 0
    leaves = [1,2,3,4]
    edges_star = [(center, leaf) for leaf in leaves]
    g2 = Graph(edges=edges_star)
    print("\nStar graph")
    perm2, score2 = random_restarts_hill_climb(g2, restarts=40)
    print_solution(perm2, g2)

