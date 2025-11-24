#include <stdlib.h>
#include <stdio.h>

int main(void) {
    int *data = malloc(100 * sizeof(int));
    if (!data) { perror("malloc"); return 1; }
    printf("Allocated array of 100 ints at %p\n", data);
    data[100] = 0; // out-of-bounds: valid indexes 0..99
    printf("Wrote data[100]=0\n");
    free(data);
    return 0;
}
