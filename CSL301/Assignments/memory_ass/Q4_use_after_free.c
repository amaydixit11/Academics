#include <stdlib.h>
#include <stdio.h>

int main(void) {
    int *arr = malloc(10 * sizeof(int));
    if (!arr) { perror("malloc"); return 1; }
    arr[0] = 42;
    printf("arr[0] before free = %d\n", arr[0]);
    free(arr);
    printf("After free: reading arr[0] -> undefined behavior: ");
    printf("%d\n", arr[0]);  // use-after-free: undefined
    return 0;
}
