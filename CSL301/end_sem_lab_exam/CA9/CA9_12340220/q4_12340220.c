    #include <stdio.h>
    #include <stdlib.h>
    #include <pthread.h>
    #include <semaphore.h>
    #include <unistd.h>

    sem_t roomEmpty;
    int readers = 0;
    int shared_data = 0;
    int read_count = 0;   
    int write_count = 0;  
    pthread_mutex_t readMutex;

    void* reader(void* arg) {
        int id = *(int*)arg;
        for(int i = 0; i < 50; i++) {

            pthread_mutex_lock(&readMutex);
            readers++;
            if(readers == 1)
                sem_wait(&roomEmpty);  // block writers when first reader enters
            pthread_mutex_unlock(&readMutex);

            // Critical section — reading
            int value = shared_data;
            read_count++;
            printf("Reader %d reads: %d (readers=%d)\n", id, value, readers);

            pthread_mutex_lock(&readMutex);
            readers--;
            if(readers == 0)
                sem_post(&roomEmpty);  // last reader leaves, allow writers
            pthread_mutex_unlock(&readMutex);

            usleep(10000);
        }
        return NULL;
    }


    void* writer(void* arg) {
        int id = *(int*)arg;
        for(int i = 0; i < 20; i++) {
            sem_wait(&roomEmpty);
            
            // Writing
            shared_data++;
            write_count++;
            printf("Writer %d writes: %d\n", id, shared_data);
            usleep(50000);
            
            sem_post(&roomEmpty);
            usleep(100000);
        }
        return NULL;
    }

    int main() {
        pthread_t reader_threads[4], writer_threads[2];
        int reader_ids[4] = {1, 2, 3, 4};
        int writer_ids[2] = {1, 2};
        
        sem_init(&roomEmpty, 0, 1);
        pthread_mutex_init(&readMutex, NULL);

        printf("=== BUGGY VERSION - Watch for anomalies ===\n");
        printf("Expected behavior: Writers should NEVER run while readers>0\n\n");
        
        for(int i = 0; i < 4; i++) {
            pthread_create(&reader_threads[i], NULL, reader, &reader_ids[i]);
        }
        for(int i = 0; i < 2; i++) {
            pthread_create(&writer_threads[i], NULL, writer, &writer_ids[i]);
        }
        
        for(int i = 0; i < 4; i++) {
            pthread_join(reader_threads[i], NULL);
        }
        for(int i = 0; i < 2; i++) {
            pthread_join(writer_threads[i], NULL);
        }
        
        sem_destroy(&roomEmpty);
        pthread_mutex_destroy(&readMutex);

        printf("\n========== RESULTS ==========\n");
        printf("Final readers counter: %d (should be 0)\n", readers);
        printf("Total reads: %d\n", read_count);
        printf("Total writes: %d (expected: 40)\n", write_count);
        printf("Final shared_data: %d (expected: 40)\n", shared_data);
        
        if(readers != 0) {
            printf("\nBUG DETECTED: readers counter is corrupted!\n");
        }
        if(write_count != 40 || shared_data != 40) {
            printf("BUG DETECTED: Data corruption occurred!\n");
        }
        
        return 0;
    }

