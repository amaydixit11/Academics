#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>


typedef struct {
    int counter;
    sem_t mutex;  
} Lightswitch;

Lightswitch readSwitch;
sem_t roomEmpty;  


void lightswitch_init(Lightswitch *ls) {
    ls->counter = 0;
    sem_init(&ls->mutex, 0, 1);
}


void lightswitch_lock(Lightswitch *ls, sem_t *semaphore) {
    sem_wait(&ls->mutex);
    ls->counter++;
    if (ls->counter == 1) {
        sem_wait(semaphore);
    }
    sem_post(&ls->mutex);
}


void lightswitch_unlock(Lightswitch *ls, sem_t *semaphore) {
    sem_wait(&ls->mutex);
    ls->counter--;
    if (ls->counter == 0) {
        sem_post(semaphore);
    }
    sem_post(&ls->mutex);
}

void *reader(void *arg) {
    int id = *(int *)arg;

    // Enter critical section for readers
    lightswitch_lock(&readSwitch, &roomEmpty);

    printf("Reader %d is reading...\n", id);
    sleep(1);  // Simulate read operation
    printf("Reader %d finished reading.\n", id);

    // Exit critical section for readers
    lightswitch_unlock(&readSwitch, &roomEmpty);

    return NULL;
}


void *writer(void *arg) {
    int id = *(int *)arg;

    // Writer requires exclusive access
    sem_wait(&roomEmpty);

    printf("Writer %d is writing...\n", id);
    sleep(2);  // Simulate write operation
    printf("Writer %d finished writing.\n", id);

    sem_post(&roomEmpty);

    return NULL;
}


int main() {
    pthread_t r1, r2, w1, w2;
    int rID1 = 1, rID2 = 2;
    int wID1 = 1, wID2 = 2;


    lightswitch_init(&readSwitch);
    sem_init(&roomEmpty, 0, 1);


    pthread_create(&r1, NULL, reader, &rID1);
    pthread_create(&r2, NULL, reader, &rID2);
    pthread_create(&w1, NULL, writer, &wID1);
    pthread_create(&w2, NULL, writer, &wID2);


    pthread_join(r1, NULL);
    pthread_join(r2, NULL);
    pthread_join(w1, NULL);
    pthread_join(w2, NULL);


    sem_destroy(&roomEmpty);
    sem_destroy(&readSwitch.mutex);

    return 0;
}

