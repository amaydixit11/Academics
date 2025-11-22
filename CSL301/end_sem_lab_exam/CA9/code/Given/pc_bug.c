#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define BUFFER_SIZE 100

sem_t items;
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
        buffer[fill_ptr] = item;
        fill_ptr = (fill_ptr + 1) % BUFFER_SIZE;
        items_produced++;  
        printf("Producer %d produced: %d\n", id, item);
        sem_post(&items);
        
    }
    return NULL;
}

void* consumer(void* arg) {
    int id = *(int*)arg;
    for(int i = 0; i < 100000; i++) {
        sem_wait(&items);
        int item = buffer[use_ptr];
        use_ptr = (use_ptr + 1) % BUFFER_SIZE;
        items_consumed++;  
        printf("Consumer %d consumed: %d\n", id, item);
      
    }
    return NULL;
}

int main() {
    pthread_t prod_threads[2], cons_threads[2];
    int prod_ids[2] = {1, 2};
    int cons_ids[2] = {1, 2};
    
    sem_init(&items, 0, 0);
    
    printf("Starting Producer-Consumer (BUGGY VERSION)\n");
    
    for(int i = 0; i < 2; i++) {
        pthread_create(&prod_threads[i], NULL, producer, &prod_ids[i]);
        pthread_create(&cons_threads[i], NULL, consumer, &cons_ids[i]);
    }
    
    for(int i = 0; i < 2; i++) {
        pthread_join(prod_threads[i], NULL);
        pthread_join(cons_threads[i], NULL);
    }
    
    sem_destroy(&items);
    
    printf("\n========== FINAL RESULTS ==========\n");
    printf("Total items produced: %d\n", items_produced);
    printf("Total items consumed: %d\n", items_consumed);
    printf("Final fill_ptr: %d\n", fill_ptr);
    printf("Final use_ptr: %d\n", use_ptr);
    
    
    return 0;
}

