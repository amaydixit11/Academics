#include <stdlib.h>
#include <stdio.h>

int main(void) {
    int *arr = malloc(10 * sizeof(int));
    if (!arr) { perror("malloc"); return 1; }
    int *mid = arr + 3;
    printf("arr=%p, mid=%p\n", (void*)arr, (void*)mid);
    // Incorrect: freeing a pointer not returned by malloc
    free(mid);
    printf("Returned from free(mid) -- may abort or print this depending on libc.\n");
    return 0;
}
