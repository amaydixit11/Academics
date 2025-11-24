#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <semaphore.h>
#include <time.h>

#define NUM_READERS 5
#define NUM_WRITERS 3
#define NUM_READ_OPS 4
#define NUM_WRITE_OPS 3
#define MIN_SLEEP_TIME_READER 100
#define MAX_SLEEP_TIME_READER 300
#define MIN_SLEEP_TIME_WRITER 150
#define MAX_SLEEP_TIME_WRITER 350

sem_t roomEmpty;
pthread_mutex_t readMutex, writeMutex;
int readers = 0;
int waiting_writers = 0;
int data_value = 0;
int log_count = 0;

void* reader(void* arg) {
    int id = *(int*)arg;
    for(int i = 0; i < NUM_READ_OPS; i++) {
        usleep((rand() % (MAX_SLEEP_TIME_READER - MIN_SLEEP_TIME_READER + 1) + MIN_SLEEP_TIME_READER) * 1000);

        pthread_mutex_lock(&writeMutex);
        while (waiting_writers > 0) {
            pthread_mutex_unlock(&writeMutex);
            usleep(1000); // wait before retrying
            pthread_mutex_lock(&writeMutex);
        }
        pthread_mutex_unlock(&writeMutex);

        pthread_mutex_lock(&readMutex);
        readers++;
        if(readers == 1)
            sem_wait(&roomEmpty);
        pthread_mutex_unlock(&readMutex);

        // Critical section — reading
        log_count++;
        printf("Reader %d is reading... Active Readers = %d | Data Value = %d\n", id, readers, data_value);

        pthread_mutex_lock(&readMutex);
        readers--;
        if(readers == 0)
            sem_post(&roomEmpty);
        pthread_mutex_unlock(&readMutex);
    }
    return NULL;
}

void* writer(void* arg) {
    int id = *(int*)arg;
    for(int i = 0; i < NUM_WRITE_OPS; i++) {
        usleep((rand() % (MAX_SLEEP_TIME_WRITER - MIN_SLEEP_TIME_WRITER + 1) + MIN_SLEEP_TIME_WRITER) * 1000);

        pthread_mutex_lock(&writeMutex);
        waiting_writers++;
        pthread_mutex_unlock(&writeMutex);

        sem_wait(&roomEmpty);

        // Writing
        data_value++;
        log_count++;
        printf("Writer %d is writing... New Data Value = %d\n", id, data_value);

        sem_post(&roomEmpty);

        pthread_mutex_lock(&writeMutex);
        waiting_writers--;
        pthread_mutex_unlock(&writeMutex);
    }
    return NULL;
}

int main() {
    printf("======= 12340220 Amay Dixit Question 3 ======\n");
    srand(time(NULL));
    pthread_t reader_threads[NUM_READERS], writer_threads[NUM_WRITERS];
    int reader_ids[NUM_READERS], writer_ids[NUM_WRITERS];

    sem_init(&roomEmpty, 0, 1);
    
    pthread_mutex_init(&readMutex, NULL);
    pthread_mutex_init(&writeMutex, NULL);

    for(int i = 0; i < NUM_READERS; i++) {
        reader_ids[i] = i + 1;
        pthread_create(&reader_threads[i], NULL, reader, &reader_ids[i]);
    }

    for(int i = 0; i < NUM_WRITERS; i++) {
        writer_ids[i] = i + 1;
        pthread_create(&writer_threads[i], NULL, writer, &writer_ids[i]);
    }

    for(int i = 0; i < NUM_READERS; i++) {
        pthread_join(reader_threads[i], NULL);
    }

    for(int i = 0; i < NUM_WRITERS; i++) {
        pthread_join(writer_threads[i], NULL);
    }

    pthread_mutex_destroy(&readMutex);
    pthread_mutex_destroy(&writeMutex);
    sem_destroy(&roomEmpty);

    printf("Final Data Value = %d\n", data_value);
    printf("Final Log Count  = %d\n", log_count);

    return 0;
}

