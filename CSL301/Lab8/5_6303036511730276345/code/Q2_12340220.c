#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

sem_t aArrived, bArrived;

void* threadA(void* arg) {
    printf("A1 (Starting task)\n");
    sem_post(&aArrived);
    sem_wait(&bArrived);
    printf("A2 (Ending task)\n");
    return NULL;
}

void* threadB(void* arg) {
    printf("B1 (Starting task)\n");
    sem_post(&bArrived);
    sem_wait(&aArrived);
    printf("B2 (Ending task)\n");
    return NULL;
}

int main() {
    pthread_t tA, tB;
    sem_init(&aArrived, 0, 0);
    sem_init(&bArrived, 0, 0);

    pthread_create(&tA, NULL, threadA, NULL);
    pthread_create(&tB, NULL, threadB, NULL);

    pthread_join(tA, NULL);
    pthread_join(tB, NULL);

    sem_destroy(&aArrived);
    sem_destroy(&bArrived);

    return 0;
}
