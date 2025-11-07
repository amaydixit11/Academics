#include <bits/stdc++.h>
#include<thread>
#include <chrono>
using namespace std;
using namespace chrono;


struct Node {
    vector<vector<int>> state; // 3x3 matrix
    int g; // cost from start
    int h; // heuristic cost to goal
    int f; // total cost
    shared_ptr<Node> parent; // to reconstruct path
};

// Goal state
vector<vector<int>> goal = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 0}
};

// Directions (up, down, left, right)
vector<pair<int,int>> moves = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}
};

// Heuristic: Manhattan Distance
int manhattan(const vector<vector<int>>& state) {
    int dist = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int val = state[i][j];
            if (val != 0) {
                int goalRow = (val - 1) / 3;
                int goalCol = (val - 1) % 3;
                dist += abs(i - goalRow) + abs(j - goalCol);
            }
        }
    }
    return dist;
}

// Check if goal reached
bool isGoal(const vector<vector<int>>& state) {
    return state == goal;
}

// Print matrix nicely
void printMatrix(const vector<vector<int>>& state) {
    for (auto &row : state) {
        for (auto val : row) {
            if (val == 0) cout << "  "; // blank space
            else cout << val << " ";
        }
        cout << "\n";
    }
    cout << "-----------------\n";
}

// Compare function for priority queue (min-heap by f value)
struct Compare {
    bool operator()(const shared_ptr<Node>& a, const shared_ptr<Node>& b) const {
        return a->f > b->f;
    }
};

// A* algorithm
void solveAStar(vector<vector<int>> start) {
    auto start_time = high_resolution_clock::now();

    priority_queue<shared_ptr<Node>, vector<shared_ptr<Node>>, Compare> openList;
    set<vector<vector<int>>> closedSet;

    auto startNode = make_shared<Node>();
    startNode->state = start;
    startNode->g = 0;
    startNode->h = manhattan(start);
    startNode->f = startNode->g + startNode->h;
    startNode->parent = nullptr;
    openList.push(startNode);

    while (!openList.empty()) {
        auto current = openList.top();
        openList.pop();

        if (isGoal(current->state)) {
            // reconstruct path
            vector<shared_ptr<Node>> path;
            while (current) {
                path.push_back(current);
                current = current->parent;
            }
            reverse(path.begin(), path.end());

            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end_time - start_time);

            cout << "Solution found in " << path.size()-1 << " moves.\n\n";
            for (size_t i = 0; i < path.size(); i++) {
                cout << "Step " << i << ":\n";
                printMatrix(path[i]->state);
            }
            cout << "Execution time: " << duration.count() << " ms"
                 << " (" << duration.count() / 1000.0 << " seconds)\n";
            return;
        }

        closedSet.insert(current->state);

        // find position of blank tile (0)
        int bx, by;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (current->state[i][j] == 0) {
                    bx = i; by = j;
                }

        // explore moves
        for (auto [dx, dy] : moves) {
            int nx = bx + dx, ny = by + dy;
            if (nx >= 0 && nx < 3 && ny >= 0 && ny < 3) {
                auto newState = current->state;
                swap(newState[bx][by], newState[nx][ny]);

                if (closedSet.find(newState) != closedSet.end()) continue;

                auto neighbor = make_shared<Node>();
                neighbor->state = newState;
                neighbor->g = current->g + 1;
                neighbor->h = manhattan(newState);
                neighbor->f = neighbor->g + neighbor->h;
                neighbor->parent = current;
                openList.push(neighbor);
            }
        }
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "No solution exists.\n";
    cout << "Execution time: " << duration.count() << " ms"
         << " (" << duration.count() / 1000.0 << " seconds)\n";


    
}

int main() {
    vector<vector<int>> start = {
        {8, 6, 7},
        {2, 5, 4},
        {3, 0, 1}
    };

    solveAStar(start);
    return 0;
}
