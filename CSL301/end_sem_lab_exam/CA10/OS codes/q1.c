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
        while (turn != 0) {
            //ADD YOUR CODE HERE
        }
        printf("A ");
        fflush(stdout);
       //ADD YOUR CODE HERE
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

void *printB(void *arg) {
    for (int i = 0; i < N; i++) {
        pthread_mutex_lock(&lock);
        //ADD YOUR CODE HERE
        printf("B ");
        fflush(stdout);
        //ADD YOUR CODE HERE
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

void *printC(void *arg) {
    for (int i = 0; i < N; i++) {
        pthread_mutex_lock(&lock);
        //ADD YOUR CODE HERE
        printf("C\n");
        fflush(stdout);
        //ADD YOUR CODE HERE
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
