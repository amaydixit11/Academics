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
        int n;
        printf("Enter number of threads: ");
        scanf("%d", &n);

        pthread_t threads[n];
        int thread_nums[n];
        void* result;
        int sum = 0;

        for (int i = 0; i < n; i++) {
            thread_nums[i] = i + 1;
            if (pthread_create(&threads[i], NULL, compute_square, &thread_nums[i]) != 0) {
                perror("Failed to create thread");
                return 1;
            }
        }

        for (int i = 0; i < n; i++) {
            pthread_join(threads[i], &result);
            int value = *(int*)result;
            printf("Thread %d returned %d\n", i + 1, value);
            sum += value;
            free(result);
        }

        printf("Sum of all threads: %d\n", sum);
        return 0;
    }
