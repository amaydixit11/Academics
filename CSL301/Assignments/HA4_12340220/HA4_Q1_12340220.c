#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#define SIZE 100
#define SEGMENT_SIZE 10
#define NUM_THREADS 11

int arr[SIZE];
int partial_sums[10];

typedef struct
{
    int start;
    int end;
    int index;
} Args;

void *compute_sum(void *arg)
{
    Args *a = (Args *)arg;
    int sum = 0;
    for (int i = a->start; i < a->end; ++i)
        sum += arr[i];
    partial_sums[a->index] = sum;
    printf("Thread %d partial sum: %d\n", a->index + 1, sum);
    free(a);
    pthread_exit(NULL);
}

void *total_sum(void *arg)
{
    int sum = 0;
    for (int i = 0; i < 10; ++i)
        sum += partial_sums[i];
    int *total = malloc(sizeof(int));
    *total = sum;
    pthread_exit(total);
}

int main()
{
    pthread_t threads[NUM_THREADS];

    for (int i = 0; i < SIZE; ++i)
        arr[i] = i + 1;

    for (int i = 0; i < 10; ++i)
    {
        Args *a = malloc(sizeof(Args));
        a->start = i * SEGMENT_SIZE;
        a->end = (i + 1) * SEGMENT_SIZE;
        a->index = i;
        pthread_create(&threads[i], NULL, compute_sum, (void *)a);
    }

    for (int i = 0; i < 10; ++i)
        pthread_join(threads[i], NULL);

    pthread_create(&threads[10], NULL, total_sum, NULL);

    int *total;
    pthread_join(threads[10], (void **)&total);

    printf("Total sum: %d\n", *total);
    free(total);
    return 0;
}
