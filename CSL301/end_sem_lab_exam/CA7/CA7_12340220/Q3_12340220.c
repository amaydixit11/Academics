    #include <stdio.h>
    #include <pthread.h>

    long long counter = 0;

    void* increment_counter(void* arg) {
        for (int i = 0; i < 10000000; i++) {
            counter++;
        }
        return NULL;
    }

    int main() {
        int NUM_THREADS = 10;
        pthread_t threads[NUM_THREADS];

        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_create(&threads[i], NULL, increment_counter, NULL);
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }

        printf("Final counter value: %lld\n", counter);
        printf("Expected value: %lld\n", (long long)NUM_THREADS * 10000000);

        return 0;
    }
