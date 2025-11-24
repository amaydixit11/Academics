#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#define SIZE 100
#define SEGMENT_SIZE 10
#define NUM_THREADS 11

int arr[SIZE];
int segment_max[10];

typedef struct
{
    int start;
    int end;
    int index;
} Args;

void *find_max(void *arg)
{
    Args *a = (Args *)arg;
    int max = arr[a->start];
    for (int i = a->start + 1; i < a->end; ++i)
        if (arr[i] > max)
            max = arr[i];
    segment_max[a->index] = max;
    printf("Thread %d max: %d\n", a->index + 1, max);
    free(a);
    pthread_exit(NULL);
}

void *find_overall_max(void *arg)
{
    int max = segment_max[0];
    for (int i = 1; i < 10; ++i)
        if (segment_max[i] > max)
            max = segment_max[i];
    int *result = malloc(sizeof(int));
    *result = max;
    pthread_exit(result);
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
        pthread_create(&threads[i], NULL, find_max, (void *)a);
    }

    for (int i = 0; i < 10; ++i)
        pthread_join(threads[i], NULL);

    pthread_create(&threads[10], NULL, find_overall_max, NULL);

    int *overall_max;
    pthread_join(threads[10], (void **)&overall_max);

    printf("Overall max: %d\n", *overall_max);
    free(overall_max);
    return 0;
}
