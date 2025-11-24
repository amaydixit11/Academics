#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

#define N 5

int count = 0;
pthread_mutex_t mutex;
sem_t turnstile1, turnstile2;

void* thread_func(void* arg) {
    int id = *(int*)arg;

    // Arrive at the barrier
    printf("Thread %d reached the barrier\n", id);

    pthread_mutex_lock(&mutex);
    count++;
    if (count == N) {
        sem_wait(&turnstile2);
        sem_post(&turnstile1);
    }
    pthread_mutex_unlock(&mutex);

    sem_wait(&turnstile1);
    sem_post(&turnstile1);

    // Passed barrier
    printf("Thread %d passed the barrier\n", id);

    pthread_mutex_lock(&mutex);
    count--;
    if (count == 0) {
        sem_wait(&turnstile1);
        sem_post(&turnstile2);
    }
    pthread_mutex_unlock(&mutex);

    sem_wait(&turnstile2);
    sem_post(&turnstile2);

    return NULL;
}

int main() {
    pthread_t threads[N];
    int ids[N];

    pthread_mutex_init(&mutex, NULL);
    sem_init(&turnstile1, 0, 0);
    sem_init(&turnstile2, 0, 1);

    for (int i = 0; i < N; i++) {
        ids[i] = i + 1;
        pthread_create(&threads[i], NULL, thread_func, &ids[i]);
    }

    for (int i = 0; i < N; i++)
        pthread_join(threads[i], NULL);

    pthread_mutex_destroy(&mutex);
    sem_destroy(&turnstile1);
    sem_destroy(&turnstile2);

    return 0;
}
