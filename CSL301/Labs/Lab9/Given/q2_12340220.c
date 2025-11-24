    #include <stdio.h>
    #include <stdlib.h>
    #include <pthread.h>
    #include <semaphore.h>
    #include <unistd.h>

    #define BUFFER_SIZE 100

    sem_t empty, full;
    pthread_mutex_t mutex;

    int buffer[BUFFER_SIZE];
    int fill_ptr = 0;
    int use_ptr = 0;
    int item_counter = 0;
    int items_produced = 0;
    int items_consumed = 0;

    int produce_item() {
        return item_counter++;
    }

    void put(int item) {
        buffer[fill_ptr] = item;
        fill_ptr = (fill_ptr + 1) % BUFFER_SIZE;
        items_produced++;
    }

    int get() {
        int item = buffer[use_ptr];
        use_ptr = (use_ptr + 1) % BUFFER_SIZE;
        items_consumed++;
        return item;
    }

    void* producer(void* arg) {
        int id = *(int*)arg;
        for(int i = 0; i < 100000; i++) {
            int item = produce_item();

            sem_wait(&empty);      // wait for an empty slot

            pthread_mutex_lock(&mutex);
            put(item);
            pthread_mutex_unlock(&mutex);

            sem_post(&full);       // signal one more full slot
        }
        return NULL;
    }

    void* consumer(void* arg) {
        int id = *(int*)arg;
        for(int i = 0; i < 100000; i++) {
            sem_wait(&full);       // wait until an item exists

            pthread_mutex_lock(&mutex);
            int item = get();
            pthread_mutex_unlock(&mutex);

            sem_post(&empty);      // signal one more empty slot
        }
        return NULL;
    }

    int main() {
        pthread_t prod_threads[2], cons_threads[2];
        int prod_ids[2] = {1, 2};
        int cons_ids[2] = {1, 2};

        pthread_mutex_init(&mutex, NULL);
        sem_init(&empty, 0, BUFFER_SIZE); // initially all slots empty
        sem_init(&full, 0, 0);            // initially no items

        printf("Starting Producer-Consumer (BOUNDED BUFFER VERSION)\n");

        for(int i = 0; i < 2; i++) {
            pthread_create(&prod_threads[i], NULL, producer, &prod_ids[i]);
            pthread_create(&cons_threads[i], NULL, consumer, &cons_ids[i]);
        }

        for(int i = 0; i < 2; i++) {
            pthread_join(prod_threads[i], NULL);
            pthread_join(cons_threads[i], NULL);
        }

        pthread_mutex_destroy(&mutex);
        sem_destroy(&empty);
        sem_destroy(&full);

        printf("\n========== FINAL RESULTS ==========\n");
        printf("Total items produced: %d\n", items_produced);
        printf("Total items consumed: %d\n", items_consumed);
        printf("Final fill_ptr: %d\n", fill_ptr);
        printf("Final use_ptr: %d\n", use_ptr);

        return 0;
    }
