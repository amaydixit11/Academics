#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#define N 102  // number of cycles

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

int turn = 0; // 0 -> A, 1 -> B, 2 -> C

void *printA(void *arg) {
    for (int i = 0; i < N; i++) {
        pthread_mutex_lock(&lock);

        while (turn != 0)
            pthread_cond_wait(&cond, &lock);

        printf("A ");
        fflush(stdout);

        turn = 1;
        pthread_cond_broadcast(&cond);

        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

void *printB(void *arg) {
    for (int i = 0; i < N; i++) {
        pthread_mutex_lock(&lock);

        while (turn != 1)
            pthread_cond_wait(&cond, &lock);

        printf("B ");
        fflush(stdout);

        turn = 2;
        pthread_cond_broadcast(&cond);

        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

void *printC(void *arg) {
    for (int i = 0; i < N; i++) {
        pthread_mutex_lock(&lock);

        while (turn != 2)
            pthread_cond_wait(&cond, &lock);

        printf("C\n");
        fflush(stdout);

        turn = 0;
        pthread_cond_broadcast(&cond);

        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

int main() {
    pthread_t tA, tB, tC;

    pthread_create(&tA, NULL, printA, NULL);
    pthread_create(&tB, NULL, printB, NULL);
    pthread_create(&tC, NULL, printC, NULL);

    pthread_join(tA, NULL);
    pthread_join(tB, NULL);
    pthread_join(tC, NULL);
    return 0;
}
