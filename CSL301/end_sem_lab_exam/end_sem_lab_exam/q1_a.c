#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#define N 4
#define NUM_DECREMENTS 10
#define DECREMENT_VALUE 10
#define THRESHOLD 50

int critical_limit = 100;
pthread_mutex_t mutex;


void* decrement(void* arg) {
    int thread_id = *((int*)arg);
    for (int i = 0; i < NUM_DECREMENTS; i++) {
        if (critical_limit > THRESHOLD) {
            critical_limit -= DECREMENT_VALUE;
            printf("Thread %d decremented. New Value: %d\n", thread_id, critical_limit);
        }
    }
    return NULL;
}

int main() {
    printf("======= 12340220 Amay Dixit Question 1a ======\n");
    srand(time(NULL));

    printf("-- starting part 1 (no mutex) --\n");
    printf("initial value : %d\n", critical_limit);

    pthread_t threads[N];
    int thread_ids[N];

    for (int i = 0; i < N; i++) {
        thread_ids[i] = i + 1;
        pthread_create(&threads[i], NULL, decrement, &thread_ids[i]);
    }

    for (int i = 0; i < N; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("final value (unsafe) : %d\n", critical_limit);

    return 0;
}