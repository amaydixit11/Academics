#include <stdlib.h>
#include <stdio.h>

int main(void) {
    size_t n = 1024 * 1024;
    void *p = malloc(n);
    if (!p) {
        perror("malloc");
        return 1;
    }
    printf("Allocated %zu bytes at %p and exiting without free\n", n, p);
    // intentionally not freeing
    return 0;
}
