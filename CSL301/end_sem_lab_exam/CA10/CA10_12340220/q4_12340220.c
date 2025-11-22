#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#define NTHREADS 3

struct thread_event {
    pthread_mutex_t m;
    pthread_cond_t c;
    int awake;
};

struct thread_event events[NTHREADS];

void *sleeper(void *id) {
    long tid = (long)id;
    pthread_mutex_lock(&events[tid].m);

    while (!events[tid].awake) {
        printf("Thread %ld sleeping...\n", tid);
        pthread_cond_wait(&events[tid].c, &events[tid].m);
    }

    printf("Thread %ld woke up!\n", tid);

    pthread_mutex_unlock(&events[tid].m);
    return NULL;
}

void *waker(void *arg) {
    sleep(2);

    printf("Waker: waking all threads...\n");

    for (int i = 0; i < NTHREADS; i++) {
        pthread_mutex_lock(&events[i].m);
        events[i].awake = 1;
        pthread_cond_signal(&events[i].c);  // wake each thread
        pthread_mutex_unlock(&events[i].m);
    }

    return NULL;
}


int main() {
    pthread_t t[NTHREADS], w;

    for (int i = 0; i < NTHREADS; i++) {
        events[i].awake = 0;
        pthread_mutex_init(&events[i].m, NULL);
        pthread_cond_init(&events[i].c, NULL);
        pthread_create(&t[i], NULL, sleeper, (void *)(long)i);
    }

    pthread_create(&w, NULL, waker, NULL);
    pthread_join(w, NULL);

    for (int i = 0; i < NTHREADS; i++)
        pthread_join(t[i], NULL);

    printf("All threads finished.\n");
    return 0;
}