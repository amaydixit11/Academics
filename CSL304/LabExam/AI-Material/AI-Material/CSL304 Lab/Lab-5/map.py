# Map Coloring Problem for India with three approaches:
# 1) Plain Backtracking
# 2) Backtracking + MRV + LCV
# 3) Backtracking + MRV + LCV + AC-3
#
# Expected outputs:
# - Final color assignment mapping for each algorithm
# - MRV selections and LCV orders per decision step (for algorithms 2 and 3)
# - A table showing the number of steps for each algorithm
#
# Note on visualization: The assignment allows either a printed mapping *or* a graphical visualization.
# Here we print the mapping and detailed logs as requested. If you want a graph plot too, say the word,
# and I'll add it.

from collections import deque, defaultdict
import pandas as pd

# -------------------------
# Problem definition (from the provided PDF)
# -------------------------

COLORS = ["Red", "Green", "Blue", "Yellow"]

adjacency = {
    "AP": ["TS", "OD", "TN", "KA"],
    "AR": ["AS", "NL"],
    "AS": ["AR", "NL", "ML", "TR", "MZ", "MN", "WB"],
    "BR": ["UP", "JH", "WB"],
    "CG": ["UP", "JH", "OD", "MH"],
    "GA": ["MH", "KA"],
    "GJ": ["MH", "RJ"],
    "HR": ["PB", "HP", "UK", "RJ"],
    "HP": ["JK", "PB", "HR", "UK"],
    "JH": ["BR", "UP", "CG", "OD", "WB"],
    "KA": ["MH", "AP", "TS", "TN", "KL", "GA"],
    "KL": ["KA", "TN"],
    "MP": ["RJ", "UP", "CG", "MH", "GJ"],
    "MH": ["GJ", "MP", "CG", "TS", "KA", "GA"],
    "MN": ["AS", "MZ", "NL"],
    "ML": ["AS", "TR"],
    "MZ": ["AS", "MN", "TR"],
    "NL": ["AR", "AS", "MN"],
    "OD": ["WB", "JH", "CG", "AP", "TS"],
    "PB": ["JK", "HP", "HR", "RJ"],
    "RJ": ["PB", "HR", "MP", "GJ", "UP"],
    "SK": ["WB"],
    "TN": ["AP", "KA", "KL"],
    "TS": ["MH", "KA", "AP", "OD"],
    "TR": ["AS", "ML", "MZ"],
    "UP": ["UK", "HR", "RJ", "MP", "CG", "JH", "BR"],
    "UK": ["HP", "HR", "UP"],
    "WB": ["BR", "JH", "OD", "AS", "SK"],
    "JK": ["HP", "PB"]
}

# Ensure undirected symmetry
for s, neigh in list(adjacency.items()):
    for n in neigh:
        adjacency.setdefault(n, [])
        if s not in adjacency[n]:
            adjacency[n].append(s)

variables = sorted(adjacency.keys())  # fixed order for plain backtracking


# -------------------------
# Utility functions
# -------------------------

def is_consistent(var, value, assignment, adj):
    """Check if assigning value to var violates any neighbor constraints."""
    for n in adj[var]:
        if n in assignment and assignment[n] == value:
            return False
    return True


def count_conflicts(var, value, assignment, adj):
    """How many neighbors would be conflicted by assigning value to var? (for LCV)"""
    conflicts = 0
    for n in adj[var]:
        if n in assignment and assignment[n] == value:
            conflicts += 1
    return conflicts


# -------------------------
# 1) Plain Backtracking
# -------------------------

def bt_plain(variables, adj):
    steps = {"nodes_visited": 0, "assignments_made": 0}
    assignment = {}

    def backtrack(idx):
        steps["nodes_visited"] += 1
        if idx == len(variables):
            return True

        var = variables[idx]
        for color in COLORS:
            if is_consistent(var, color, assignment, adj):
                assignment[var] = color
                steps["assignments_made"] += 1
                if backtrack(idx + 1):
                    return True
                del assignment[var]
        return False

    success = backtrack(0)
    return success, assignment, steps


# -------------------------
# 2) Backtracking + MRV + LCV
# -------------------------

def select_unassigned_var_mrv(domains, assignment, adj):
    """MRV: pick var with the smallest domain size (tie-break: highest degree)."""
    unassigned = [v for v in domains if v not in assignment]
    # domain size
    mrv_vars = sorted(unassigned, key=lambda v: len(domains[v]))
    min_size = len(domains[mrv_vars[0]])
    candidates = [v for v in mrv_vars if len(domains[v]) == min_size]
    if len(candidates) == 1:
        return candidates[0]
    # tie-break by degree (more neighbors first)
    return max(candidates, key=lambda v: len(adj[v]))


def order_values_lcv(var, domains, assignment, adj):
    """LCV: order values that eliminate the fewest options for neighbors first."""
    # For each color, count the number of ruled-out values in neighbors' domains
    scores = []
    for color in domains[var]:
        eliminated = 0
        for n in adj[var]:
            if n in assignment:
                continue
            if color in domains[n]:
                eliminated += 1
        scores.append((color, eliminated))
    # least constraining first
    ordered = [c for c, _ in sorted(scores, key=lambda x: x[1])]
    return ordered, scores


def bt_mrv_lcv(adj):
    steps = {"nodes_visited": 0, "assignments_made": 0}
    assignment = {}
    domains = {v: set(COLORS) for v in adj}
    mrv_log = []  # to record MRV selections
    lcv_log = []  # to record LCV order per step

    def backtrack():
        steps["nodes_visited"] += 1
        if len(assignment) == len(adj):
            return True

        var = select_unassigned_var_mrv(domains, assignment, adj)
        mrv_log.append(var)

        ordered_vals, lcv_scores = order_values_lcv(var, domains, assignment, adj)
        lcv_log.append((var, ordered_vals, lcv_scores))

        for color in ordered_vals:
            if is_consistent(var, color, assignment, adj):
                # forward checking lite: just assign
                assignment[var] = color
                steps["assignments_made"] += 1
                # proceed
                if backtrack():
                    return True
                del assignment[var]
        return False

    success = backtrack()
    return success, assignment, steps, mrv_log, lcv_log


# -------------------------
# 3) Backtracking + MRV + LCV + AC-3
# -------------------------

def ac3(domains, adj):
    """AC-3 for arc consistency. Returns False if a domain is wiped out."""
    queue = deque()
    for xi in domains:
        for xj in adj[xi]:
            queue.append((xi, xj))

    def revise(xi, xj):
        revised = False
        to_remove = set()
        for x in domains[xi]:
            # x is supported if exists y in domain[xj] with y != x
            if all(x == y for y in domains[xj]):
                # all y equal x -> if domain[xj] == {x}, then xi cannot be x
                if len(domains[xj]) == 1 and x in domains[xj]:
                    to_remove.add(x)
            # Another (more standard) way:
            # if no y in domain[xj] such that y != x, remove x.
            if not any(y != x for y in domains[xj]):
                to_remove.add(x)
        if to_remove:
            domains[xi] -= to_remove
            revised = True
        return revised

    while queue:
        xi, xj = queue.popleft()
        if revise(xi, xj):
            if not domains[xi]:
                return False
            for xk in adj[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True


def bt_mrv_lcv_ac3(adj):
    steps = {"nodes_visited": 0, "assignments_made": 0}
    assignment = {}
    # initialize domains
    domains = {v: set(COLORS) for v in adj}
    mrv_log = []
    lcv_log = []

    def backtrack():
        steps["nodes_visited"] += 1
        if len(assignment) == len(adj):
            return True

        var = select_unassigned_var_mrv(domains, assignment, adj)
        mrv_log.append(var)

        ordered_vals, lcv_scores = order_values_lcv(var, domains, assignment, adj)
        lcv_log.append((var, ordered_vals, lcv_scores))

        for color in ordered_vals:
            if is_consistent(var, color, assignment, adj):
                # Save current domains to restore after trying this value
                snapshot = {v: set(domains[v]) for v in domains}
                assignment[var] = color
                steps["assignments_made"] += 1

                # Enforce assignment in domains
                domains[var] = {color}
                # Run AC-3
                ok = ac3(domains, adj)
                if ok:
                    # Also remove color from neighbors to reflect constraint propagation
                    success = backtrack()
                    if success:
                        return True

                # Restore
                assignment.pop(var, None)
                for v in domains:
                    domains[v] = set(snapshot[v])
        return False

    # Initial AC-3 (optional, but can help)
    ac3(domains, adj)
    success = backtrack()
    return success, assignment, steps, mrv_log, lcv_log


# -------------------------
# Run all three and print required outputs
# -------------------------

print("==== 1) Plain Backtracking ====")
ok1, assign1, steps1 = bt_plain(variables, adjacency)
print("Success:", ok1)
print("Final Assignment:")
print(assign1)
print("Steps:", steps1)
print()

print("==== 2) Backtracking + MRV + LCV ====")
ok2, assign2, steps2, mrv2, lcv2 = bt_mrv_lcv(adjacency)
print("Success:", ok2)
print("Final Assignment:")
print(assign2)
print("Steps:", steps2)
print("\nMRV selections in order:")
print(mrv2)
print("\nLCV orders per decision step:")
for var, ordered_vals, scores in lcv2:
    print(f"Var={var} | LCV order={ordered_vals} | (value, eliminated_for_neighbors)={[(v,e) for v,e in scores]}")
print()

print("==== 3) Backtracking + MRV + LCV + AC-3 ====")
ok3, assign3, steps3, mrv3, lcv3 = bt_mrv_lcv_ac3(adjacency)
print("Success:", ok3)
print("Final Assignment:")
print(assign3)
print("Steps:", steps3)
print("\nMRV selections in order:")
print(mrv3)
print("\nLCV orders per decision step:")
for var, ordered_vals, scores in lcv3:
    print(f"Var={var} | LCV order={ordered_vals} | (value, eliminated_for_neighbors)={[(v,e) for v,e in scores]}")

# -------------------------
# Comparison table
# -------------------------

df = pd.DataFrame([
    {"Algorithm": "Plain Backtracking", "Nodes Visited": steps1["nodes_visited"], "Assignments": steps1["assignments_made"]},
    {"Algorithm": "MRV + LCV", "Nodes Visited": steps2["nodes_visited"], "Assignments": steps2["assignments_made"]},
    {"Algorithm": "MRV + LCV + AC-3", "Nodes Visited": steps3["nodes_visited"], "Assignments": steps3["assignments_made"]},
])

df

# Also save final outputs to files for convenience
import json, os

outputs = {
    "plain": {"success": ok1, "assignment": assign1, "steps": steps1},
    "mrv_lcv": {"success": ok2, "assignment": assign2, "steps": steps2, "mrv": mrv2, "lcv": lcv2},
    "mrv_lcv_ac3": {"success": ok3, "assignment": assign3, "steps": steps3, "mrv": mrv3, "lcv": lcv3},
}

os.makedirs("/mnt/data/results", exist_ok=True)
with open("/mnt/data/results/assignments_and_logs.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("\nSaved detailed results to /mnt/data/results/assignments_and_logs.json")
