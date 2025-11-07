import heapq

start_state = ((1, 2, 3),
               (4, 0, 6),
               (7, 5, 8))

goal_state = ((1, 2, 3),
              (4, 5, 6),
              (7, 8, 0))

def manhattan_distance(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:
                goal_i = (value - 1) // 3
                goal_j = (value - 1) % 3
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance

def get_neighbors(state):
    neighbors = []
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                x, y = i, j
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [list(row) for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append(tuple(tuple(row) for row in new_state))
    return neighbors

def astar(start, goal):
    open_set = []
    heapq.heappush(open_set, (manhattan_distance(start), 0, start))
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan_distance(neighbor)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    return None

def print_state(state):
    for row in state:
        print(" ".join(str(x) if x != 0 else " " for x in row))
    print()

path = astar(start_state, goal_state)
if path:
    for step, state in enumerate(path):
        print(f"Step {step}:")
        print_state(state)
else:
    print("No solution found.")

