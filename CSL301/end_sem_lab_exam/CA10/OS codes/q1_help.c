/*Running two threads alternately
  pthread_cond_signal- for awaking one thread
  pthread_cond_broadcast -for awaking all threads
*/
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

int turn = 0;  // 0 = A's turn, 1 = B's turn
int cycles = 100; // number of times to print A-B

void *printA(void *arg) {
    for (int i = 0; i < cycles; i++) {
        pthread_mutex_lock(&lock);

        // Wait until it's A's turn
        while (turn != 0)
            pthread_cond_wait(&cond, &lock);

        printf("A ");
        fflush(stdout);

        // Switch to B's turn
        turn = 1;
        pthread_cond_signal(&cond);

        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

void *printB(void *arg) {
    for (int i = 0; i < cycles; i++) {
        pthread_mutex_lock(&lock);

        // Wait until it's B's turn
        while (turn != 1)
            pthread_cond_wait(&cond, &lock);

        printf("B ");
        fflush(stdout);

        // Switch to A's turn
        turn = 0;
        pthread_cond_signal(&cond);

        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

int main() {
    pthread_t tA, tB;

    pthread_create(&tA, NULL, printA, NULL);
    pthread_create(&tB, NULL, printB, NULL);

    pthread_join(tA, NULL);
    pthread_join(tB, NULL);

    printf("\nDone!\n");
    return 0;
}

