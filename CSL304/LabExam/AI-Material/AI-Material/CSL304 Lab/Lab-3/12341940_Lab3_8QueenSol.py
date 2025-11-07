import random
import math

def count_conflicts(board):
    size = len(board)
    conflicts = 0
    for col1 in range(size):
        for col2 in range(col1 + 1, size):
            if board[col1] == board[col2] or abs(board[col1] - board[col2]) == abs(col1 - col2):
                conflicts += 1
    return conflicts

def get_random_neighbor(board):
    size = len(board)
    new_board = board[:]
    col = random.randint(0, size - 1)
    row = random.randint(0, size - 1)
    while row == new_board[col]:
        row = random.randint(0, size - 1)
    new_board[col] = row
    return new_board

def solve_n_queens(board_size=8, max_steps=100000, start_temp=100, cooling=0.99):
    current_board = [random.randint(0, board_size - 1) for _ in range(board_size)]
    current_conflicts = count_conflicts(current_board)
    temperature = start_temp

    for step in range(max_steps):
        if current_conflicts == 0:  
            return current_board, step

        next_board = get_random_neighbor(current_board)
        next_conflicts = count_conflicts(next_board)

        delta = next_conflicts - current_conflicts

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_board = next_board
            current_conflicts = next_conflicts

        temperature *= cooling
        if temperature < 1e-6:  
            break

    return current_board, current_conflicts


solution, result = solve_n_queens()
print("Final Board:", solution)
print("Conflicts:", result)

