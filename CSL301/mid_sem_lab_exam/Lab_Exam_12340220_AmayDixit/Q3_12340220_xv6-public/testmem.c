#include "types.h"
#include "stat.h"
#include "user.h"

int main(int argc, char *argv[]) {
    
    // Your test code here
    char *result = memlayout();
    
    printf(1, "stack address: %p\n", (void*)&result);
    exit();
}