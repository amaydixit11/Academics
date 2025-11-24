#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>


void* thread_function(void* arg) {
    int thread_num = *(int*)arg;
    printf("Thread %d running\n", thread_num);
    return NULL;
}

int main() {
    int NUM_THREADS = 10;
    pthread_t threads[NUM_THREADS];
    int thread_nums[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_nums[i] = i;
        if (pthread_create(&threads[i], NULL, thread_function, &thread_nums[i]) != 0) {
            perror("Failed to create thread");
            exit(1);
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("All threads completed\n");
    return 0;
}
