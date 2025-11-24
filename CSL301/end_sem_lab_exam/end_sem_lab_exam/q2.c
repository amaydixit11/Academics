#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define ARRAY_SIZE 1000
#define NUM_THREADS 4

typedef struct {
    int min;
    int max;
} MinMax;

typedef struct {
    int* array;
    int start_index;
    int end_index;
} ThreadData;

void* find_min_max(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int min = data->array[data->start_index];
    int max = data->array[data->start_index];
    for (int i = data->start_index + 1; i < data->end_index; i++) {
        if (data->array[i] < min) {
            min = data->array[i];
        }
        if (data->array[i] > max) {
            max = data->array[i];
        }
    }
    MinMax* result = (MinMax*)malloc(sizeof(MinMax));
    result->min = min;
    result->max = max;
    return (void*)result;
}

int main() {
    printf("======= 12340220 Amay Dixit Question 2 ======\n");
    int array[ARRAY_SIZE];
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    MinMax* thread_results[NUM_THREADS];

    srand(time(NULL));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 10000; 
    }
    
    int segment_size = ARRAY_SIZE / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].array = array;
        thread_data[i].start_index = i * segment_size;
        thread_data[i].end_index = (i == NUM_THREADS - 1) ? ARRAY_SIZE : (i + 1) * segment_size;
        pthread_create(&threads[i], NULL, find_min_max, (void*)&thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], (void**)&thread_results[i]);
    }
    
    int overall_min = thread_results[0]->min;
    int overall_max = thread_results[0]->max;
    for (int i = 1; i < NUM_THREADS; i++) {
        if (thread_results[i]->min < overall_min) {
            overall_min = thread_results[i]->min;
        }
        if (thread_results[i]->max > overall_max) {
            overall_max = thread_results[i]->max;
        }
        free(thread_results[i]); 
    }
    free(thread_results[0]); 

    
    printf("Overall Minimum Value: %d\n", overall_min);
    printf("Overall Maximum Value: %d\n", overall_max);

    return 0;
}