#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void* compute_square(void* arg) {
    int n = *(int*)arg;
    int* result = malloc(sizeof(int));
    *result = n * n;
    return result;
}

int main() {
    pthread_t tid;
    int n;

    printf("Enter a number: ");
    scanf("%d", &n);
    if (pthread_create(&tid, NULL, compute_square, &n) != 0) {
        perror("Failed to create thread");
        return 1;
    }
    void* result;
    pthread_join(tid, &result);
    printf("Square of %d is %d\n", n, *(int*)result);
    free(result);

    return 0;
}
