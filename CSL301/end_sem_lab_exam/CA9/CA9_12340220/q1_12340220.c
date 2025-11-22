    #include <stdio.h>
    #include <stdlib.h>
    #include <pthread.h>
    #include <semaphore.h>
    #include <unistd.h>

    #define BUFFER_SIZE 100

    sem_t items;
    pthread_mutex_t mutex;  // Mutex for shared data

    int buffer[BUFFER_SIZE];
    int fill_ptr = 0;
    int use_ptr = 0;
    int item_counter = 0;
    int items_produced = 0;
    int items_consumed = 0;

    int produce_item() {
        return item_counter++;
    }

    void* producer(void* arg) {
        int id = *(int*)arg;
        for(int i = 0; i < 100000; i++) {
            int item = produce_item();
            
            pthread_mutex_lock(&mutex);
            buffer[fill_ptr] = item;
            fill_ptr = (fill_ptr + 1) % BUFFER_SIZE;
            items_produced++;
            pthread_mutex_unlock(&mutex);

            sem_post(&items);
            printf("Producer %d produced: %d\n", id, item);
        }
        return NULL;
    }

    void* consumer(void* arg) {
        int id = *(int*)arg;
        for(int i = 0; i < 100000; i++) {
            sem_wait(&items);

            pthread_mutex_lock(&mutex);
            int item = buffer[use_ptr];
            use_ptr = (use_ptr + 1) % BUFFER_SIZE;
            items_consumed++;
            pthread_mutex_unlock(&mutex);

            printf("Consumer %d consumed: %d\n", id, item);
        }
        return NULL;
    }

    int main() {
        pthread_t prod_threads[2], cons_threads[2];
        int prod_ids[2] = {1, 2};
        int cons_ids[2] = {1, 2};

        sem_init(&items, 0, 0);
        pthread_mutex_init(&mutex, NULL);

        printf("Starting Producer-Consumer (FIXED VERSION)\n");

        for(int i = 0; i < 2; i++) {
            pthread_create(&prod_threads[i], NULL, producer, &prod_ids[i]);
            pthread_create(&cons_threads[i], NULL, consumer, &cons_ids[i]);
        }

        for(int i = 0; i < 2; i++) {
            pthread_join(prod_threads[i], NULL);
            pthread_join(cons_threads[i], NULL);
        }

        sem_destroy(&items);
        pthread_mutex_destroy(&mutex);

        printf("\n========== FINAL RESULTS ==========\n");
        printf("Total items produced: %d\n", items_produced);
        printf("Total items consumed: %d\n", items_consumed);
        printf("Final fill_ptr: %d\n", fill_ptr);
        printf("Final use_ptr: %d\n", use_ptr);

        return 0;
    }
