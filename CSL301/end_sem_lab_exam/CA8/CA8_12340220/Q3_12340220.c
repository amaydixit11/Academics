#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 10
#define INCREMENTS 100

long counter = 0;
pthread_mutex_t lock;

void *increment_without_mutex(void *arg)
{
    for (int i = 0; i < INCREMENTS / NUM_THREADS; i++){
        if (counter > 50) counter -= 10;
    }
    return NULL;
}

void *increment_with_mutex(void *arg)
{
    for (int i = 0; i < INCREMENTS / NUM_THREADS; i++)
    {
        pthread_mutex_lock(&lock);
        if (counter > 50) counter -= 10;
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

int main()
{
    pthread_t threads[NUM_THREADS];

    // Part 1: Without Mutex
    counter = 1000;
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, increment_without_mutex, NULL);

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    printf("Final Counter (without mutex): %ld\n", counter);

    // Part 2: With Mutex
    counter = 1000;
    pthread_mutex_init(&lock, NULL);

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, increment_with_mutex, NULL);

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    printf("Final Counter (with mutex): %ld\n", counter);

    pthread_mutex_destroy(&lock);
    return 0;
}
