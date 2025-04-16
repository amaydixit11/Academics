#include <iostream>
#include <string>
#include <vector>

using namespace std;

enum CacheState {
    INVALID,
    SHARED,
    EXCLUSIVE,
    MODIFIED
};

string toString(CacheState state) {
    switch (state) {
    case INVALID:
        return "INVALID";
    case SHARED:
        return "SHARED";
    case EXCLUSIVE:
        return "EXCLUSIVE";
    case MODIFIED:
        return "MODIFIED";
    default:
        return "UNKNOWN";
    }
}

class CacheSystem {
private:
    vector<CacheState> caches_array;

public:
    CacheSystem() : caches_array(4, INVALID) {}

    void read(int process) {
        if (caches_array[process] == INVALID) {
            bool hasData = false;
            bool isModified = false;

            for (int i = 0; i < 4; ++i) {
                if (i != process && (caches_array[i] == SHARED || caches_array[i] == EXCLUSIVE || caches_array[i] == MODIFIED)) {
                    hasData = true;
                    if (caches_array[i] == MODIFIED) isModified = true;   
                }
            }

            if (hasData){
                caches_array[process] = SHARED;

                for (int i = 0; i < 4; ++i){
                    if (caches_array[i] == MODIFIED){
                        caches_array[i] = SHARED;
                    }
                    else if (caches_array[i] == EXCLUSIVE) {
                        caches_array[i] = SHARED;
                    }
                }
            }
            else caches_array[process] = EXCLUSIVE;
            
        }
    }

    void write(int process) {
        for (int i = 0; i < 4; ++i) {
            if (i != process) caches_array[i] = INVALID;
        }

        caches_array[process] = MODIFIED;
    }

    void printAnswer() {
        for (int i = 0; i < 4; ++i) {
            cout << "C" << (i + 1) << ": " << toString(caches_array[i]);
            if (i < 3)
                cout << ", ";
        }
        cout << endl;
    }
};

int main() {
    CacheSystem system;
    int a;
    char b;
    int c;

    while (true) {
        cout << "Enter (a b c) : ";
        cin >> a >> b >> c;

        if (a < 1 || a > 4 || (b != 'R' && b != 'W') || (c != 0 && c != 1)) {
            cout << "Please enter a valid input. Format: a b c (a ∈ {1,2,3,4}, b ∈ {R,W}, c ∈ {0,1})" << endl;
            continue;
        }

        if (b == 'R') system.read(a - 1);
        else system.write(a - 1);
        
        system.printAnswer();
        if (c == 0) break;
    }
    return 0;
}